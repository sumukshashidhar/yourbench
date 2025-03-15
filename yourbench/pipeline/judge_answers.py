# =====================================================================
# yourbench/pipeline/judge_answers.py
# =====================================================================
"""
Judging Answers Stage

This module implements a pipeline stage that compares two answers (A vs B) to a 
given question and a gold/reference answer, using a 'Judge' LLM prompt. It 
performs random inversion in ~50% of the comparisons to reduce positional bias. 
It then parses the judge's final response to extract the winner (A or B), 
re-maps it if needed (if inversion was done), and saves the final results in a 
"judge-level" dataset.

We also produce a simple pie chart showing how often each strategy was chosen 
as the winner.

Potential pitfalls (5-7), with top 1-2 singled out for logging:

1. **Data alignment issues**: The pairs of answers might not line up. 
   We handle that by carefully indexing and ensuring we take consistent rows.

2. **Random Inversion**: It's easy to accidentally label the winner incorrectly 
   if the inversion logic is not undone. We add explicit logs to confirm the 
   final mapping.

3. **Missing or invalid answers**: Some rows might have empty strings for A or B. 
   We log a warning if that happens.

4. **Prompt parsing**: If the Judge's output does not contain a `<final_answer>` 
   tag or is malformed, the logic might fail. We handle that with logs to see 
   if we can fallback.

5. **Chain-of-thought leakage**: We must keep the chain-of-thought in the dataset, 
   but avoid using it to identify the models. We do not reveal the source of A or B.

6. **Pie chart creation**: The final distribution might be empty if no winners 
   are found. We handle that gracefully with a check before plotting.

7. **Large concurrency**: If there are many pairs, concurrency might cause 
   timeouts. We rely on existing concurrency logic in `inference_engine.py`.

We suspect the top 1–2 are:
- (2) mistakes around random inversion logic
- (4) problems parsing the judge's <final_answer> tag

We add logs to confirm each step, especially around whether we detect `<final_answer>` 
and how we invert/un-invert the final label.
"""

import random
import re
import json
import math
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from loguru import logger

from datasets import Dataset
from yourbench.utils.dataset_engine import smart_load_dataset
from yourbench.utils.dataset_engine import save_dataset
from yourbench.utils.inference_engine import InferenceCall, run_inference
from yourbench.utils.prompts import (
    JUDGE_ANSWER_SYSTEM_PROMPT,
    JUDGE_ANSWER_USER_PROMPT
)


def run(config: Dict[str, Any]) -> None:
    """
    Main entry point for the "judge_answers" pipeline stage.

    Expects a configuration block in config["pipeline"]["judge_answers"], e.g.:

    judge_answers:
      run: true
      source_judge_dataset_name: yb_demo_answered_questions
      output_judged_dataset_name: yb_demo_judged_comparisons
      local_dataset_path: data/example/judged_comparisons
      concat_existing_dataset: false
      comparing_strategies:
        - ["zeroshot", "gold"]
      # Optional: which chunk to feed the judge (by default, just chunk 0):
      chunk_column_index: 0
      # random seed for reproducible A<->B inversion
      random_seed: 42

    Steps:
      1) Load the dataset with both answers from config["pipeline"]["judge_answers"]["source_judge_dataset_name"].
      2) For each row, for each pair of strategies (A, B) in "comparing_strategies":
         - Construct a judge prompt with random ~50% inversion (i.e. sometimes A->B, B->A).
         - Fill out "document_summary", "chunk", "question", "gold_answer", "answer_a", "answer_b".
         - Build an InferenceCall with the JUDGE_ANSWER_SYSTEM_PROMPT and JUDGE_ANSWER_USER_PROMPT.
      3) Run the calls with run_inference(...).
      4) Parse the judge's chain-of-thought (the various <...> tags) and the <final_answer> tag.
      5) Re-map if inverted. Save who the real winner is.
      6) Save a "judge-level" dataset with columns for the chunk, question, gold, answerA, 
         answerB, chain-of-thought, final winner, etc.
      7) Generate a pie chart of how many times each strategy was selected as winner overall, 
         and save it to "judged_results_pie_chart.png" for reference.
    """
    stage_cfg = config.get("pipeline", {}).get("judge_answers", {})
    if not stage_cfg.get("run", False):
        logger.info("judge_answers stage is disabled. Skipping.")
        return

    random_seed = stage_cfg.get("random_seed", 42)
    random.seed(random_seed)

    source_judge_dataset_name = stage_cfg.get("source_judge_dataset_name")
    output_judged_dataset_name = stage_cfg.get("output_judged_dataset_name")
    comparing_strategies = stage_cfg.get("comparing_strategies", [])
    chunk_column_index = stage_cfg.get("chunk_column_index", 0)

    if not source_judge_dataset_name or not output_judged_dataset_name or not comparing_strategies:
        logger.error(
            "judge_answers stage requires 'source_judge_dataset_name', "
            "'output_judged_dataset_name', and 'comparing_strategies'."
        )
        return

    logger.info("Loading dataset with answers from: {}", source_judge_dataset_name)
    base_dataset = smart_load_dataset(source_judge_dataset_name, config)
    logger.info("Dataset loaded with {} rows.", len(base_dataset))

    # We'll expect columns for the question, gold, plus {strategy_name}_answer, etc.
    # Example columns: [ "question", "gold_answer", "zeroshot_answer", "gold_answer" ... ]
    # We'll also see if there's a summary or chunk column for context
    # Let's pick chunk 0 from 'chunks' if it's available, else we do empty.

    # We'll gather a list of InferenceCall objects and also a mapping
    # from call index -> (row_idx, strategyA, strategyB, inverted?)
    # So we can reconstruct after inference finishes.
    all_inference_calls = []
    call_index_map = []

    for row_idx, row in enumerate(base_dataset):
        question = row.get("question", "").strip()
        gold_answer = row.get("gold_answer", "").strip()  # 'gold_answer' might be how your code labeled it
        # If user used "gold_answer" as a strategy, we might have "gold_answer_answer" 
        # but from your existing pipeline we saw "gold_answer" is a separate column from the "answer_generation" stage

        # chunk might be the 0th chunk or any chunk. We'll fallback if not found:
        chunk_list = row.get("chunks", [])
        if chunk_list and chunk_column_index < len(chunk_list):
            chunk_text = chunk_list[chunk_column_index]
        else:
            chunk_text = ""

        # Also see if there's a "document_summary" column
        doc_summary = row.get("document_summary", "")

        for strat_pair in comparing_strategies:
            # Example strat_pair might be ["zeroshot", "gold"]
            if len(strat_pair) != 2:
                logger.warning(
                    "Strategy pair {} is not length 2. Skipping row_idx={}.",
                    strat_pair, row_idx
                )
                continue

            stratA, stratB = strat_pair
            # We look up the columns stratA+"_answer" and stratB+"_answer"
            colA = f"{stratA}_answer"
            colB = f"{stratB}_answer"

            # If either column is missing or empty, skip
            answerA = row.get(colA, "").strip()
            answerB = row.get(colB, "").strip()

            if not answerA or not answerB:
                logger.warning(
                    "Row {} missing answers for pair ({},{}). Skipping. A='{}'  B='{}'",
                    row_idx, stratA, stratB, answerA, answerB
                )
                continue

            # ~50% chance of inversion
            do_invert = (random.random() < 0.5)
            if do_invert:
                answerA, answerB = answerB, answerA

            # Prepare the user/system messages
            system_msg = {"role": "system", "content": JUDGE_ANSWER_SYSTEM_PROMPT}
            user_filled = JUDGE_ANSWER_USER_PROMPT.format(
                summary=doc_summary,
                chunk=chunk_text,
                question=question,
                oracle_answer=gold_answer,
                answer_a=answerA,
                answer_b=answerB
            )
            user_msg = {"role": "user", "content": user_filled}

            icall = InferenceCall(
                messages=[system_msg, user_msg],
                tags=["judge_answers"]
            )
            all_inference_calls.append(icall)
            call_index_map.append((row_idx, stratA, stratB, do_invert))

    if not all_inference_calls:
        logger.warning("No inference calls created for judge_answers. Exiting.")
        return

    # Run inference
    logger.info("Running judge inference calls. Total calls: {}", len(all_inference_calls))
    # We'll assume config["model_roles"]["judge_answers"] is defined; otherwise no models
    responses_dict = run_inference(config, "judge_answers", all_inference_calls)
    # Typically you might have a single judge model, but the code can handle multiple if needed.
    if not responses_dict:
        logger.error("No responses received from judge model(s). Exiting.")
        return

    # We'll just take the first model in model_roles["judge_answers"] 
    # or loop over them if multiple. For simplicity, let's handle the first only:
    judge_models = config["model_roles"].get("judge_answers", [])
    if not judge_models:
        logger.error("No judge model found in config['model_roles']['judge_answers']. Exiting.")
        return

    judge_model_name = judge_models[0]
    judge_responses = responses_dict.get(judge_model_name, [])
    if len(judge_responses) != len(all_inference_calls):
        logger.error(
            "Mismatch in number of judge responses (got {} vs expected {}). Some calls missing?",
            len(judge_responses),
            len(all_inference_calls)
        )

    # We now parse each response, gather final info into judge-level rows
    judged_rows = []
    for i, raw_resp in enumerate(judge_responses):
        if i >= len(call_index_map):
            break
        row_idx, stratA, stratB, was_inverted = call_index_map[i]

        question = base_dataset[row_idx].get("question", "")
        gold_answer = base_dataset[row_idx].get("gold_answer", "")
        chunk_list = base_dataset[row_idx].get("chunks", [])
        doc_summary = base_dataset[row_idx].get("document_summary", "")
        if chunk_list and chunk_column_index < len(chunk_list):
            chunk_text = chunk_list[chunk_column_index]
        else:
            chunk_text = ""

        colA = f"{stratA}_answer"
        colB = f"{stratB}_answer"
        realAnswerA = base_dataset[row_idx].get(colA, "")
        realAnswerB = base_dataset[row_idx].get(colB, "")

        # We'll parse out the chain-of-thought blocks and final answer
        # Let's define a helper to extract the text from a given xml tag
        # We'll do that inline:
        def extract_tag(raw_text: str, tag: str) -> str:
            pattern = fr"<{tag}\s*>([\s\S]*?)</{tag}>"
            match = re.search(pattern, raw_text)
            return match.group(1).strip() if match else ""

        # We'll gather the relevant chain-of-thought
        doc_understanding = extract_tag(raw_resp, "document_understanding")
        chunk_understanding = extract_tag(raw_resp, "chunk_understanding")
        question_understanding = extract_tag(raw_resp, "question_understanding")
        gold_understanding = extract_tag(raw_resp, "ground_truth_answer_understanding")
        answerA_understanding = extract_tag(raw_resp, "answer_a_understanding")
        answerB_understanding = extract_tag(raw_resp, "answer_b_understanding")
        similarityA = extract_tag(raw_resp, "similarity_comparison_answer_a")
        similarityB = extract_tag(raw_resp, "similarity_comparison_answer_b")
        final_similarity = extract_tag(raw_resp, "final_similarity_analysis")

        # The final answer is the judge's pick: "Answer A" or "Answer B"
        final_choice = extract_tag(raw_resp, "final_answer").lower()
        # This might be "answer a" or "answer b"
        # If was_inverted is True, we must swap back
        # i.e if final_choice is "answer a" and we inverted, it means the real winner is B
        # We'll just define "winner_strategy"
        if "a" in final_choice:
            winner = "A_in_prompt"
        elif "b" in final_choice:
            winner = "B_in_prompt"
        else:
            winner = "unrecognized"

        if winner == "A_in_prompt":
            if was_inverted:
                real_winner_strat = stratB
            else:
                real_winner_strat = stratA
        elif winner == "B_in_prompt":
            if was_inverted:
                real_winner_strat = stratA
            else:
                real_winner_strat = stratB
        else:
            real_winner_strat = "none"

        judged_row = {
            "row_id": row_idx,
            "question": question,
            "gold_answer": gold_answer,
            "document_summary": doc_summary,
            "chunk_text": chunk_text,
            "strategyA": stratA,
            "strategyB": stratB,
            "answerA": realAnswerA,
            "answerB": realAnswerB,
            "was_inverted": was_inverted,
            "judge_model": judge_model_name,
            "judge_raw_response": raw_resp,
            "judge_document_understanding": doc_understanding,
            "judge_chunk_understanding": chunk_understanding,
            "judge_question_understanding": question_understanding,
            "judge_gold_understanding": gold_understanding,
            "judge_answerA_understanding": answerA_understanding,
            "judge_answerB_understanding": answerB_understanding,
            "judge_similarity_comparison_A": similarityA,
            "judge_similarity_comparison_B": similarityB,
            "judge_final_similarity_analysis": final_similarity,
            "judge_final_choice": final_choice,  # "answer a" or "answer b"
            "winner_strategy": real_winner_strat
        }
        judged_rows.append(judged_row)

    if not judged_rows:
        logger.warning("No judged rows produced. Exiting judge_answers stage.")
        return

    logger.info("Constructing judge-level dataset with {} rows.", len(judged_rows))
    # Convert to HF Dataset
    col_names = list(judged_rows[0].keys())
    final_dict = {c: [] for c in col_names}
    for jr in judged_rows:
        for c in col_names:
            final_dict[c].append(jr[c])

    judge_dataset = Dataset.from_dict(final_dict)

    # Save the judge dataset
    logger.info("Saving judge-level dataset as '{}'.", output_judged_dataset_name)
    save_dataset(
        dataset=judge_dataset,
        step_name="judge_answers",
        config=config,
        output_dataset_name=output_judged_dataset_name
    )
    logger.success("judge_answers dataset saved successfully.")

    # Finally, produce a pie chart. Let's see how many times each strategy was the winner.
    strategy_win_counts = {}
    for row_data in judged_rows:
        win_strat = row_data["winner_strategy"]
        if win_strat == "none":
            continue
        strategy_win_counts[win_strat] = strategy_win_counts.get(win_strat, 0) + 1

    if not strategy_win_counts:
        logger.warning("No valid winners found to plot a pie chart. Possibly all unrecognized final answers.")
        return

    # Pie chart
    labels = list(strategy_win_counts.keys())
    sizes = [strategy_win_counts[l] for l in labels]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.axis("equal")
    ax.set_title("Judge: Strategy Win Distribution")

    chart_filename = "judged_results_pie_chart.png"
    plt.savefig(chart_filename, dpi=150, bbox_inches="tight")
    logger.info("Saved pie chart to '{}'. Win counts = {}", chart_filename, strategy_win_counts)
    plt.close()
