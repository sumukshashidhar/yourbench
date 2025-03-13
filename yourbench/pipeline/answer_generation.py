# =======================================
# yourbench/pipeline/answer_generation.py
# =======================================
"""
Answer Generation Module

This stage takes an existing question-level dataset (single-hop or multi-hop questions)
and produces model-generated answers under multiple "answering strategies" (e.g., zero-shot,
gold, retrieval-augmented, etc.). Each strategy can have a different prompt and a different model.

We then add columns to each row such that for a strategy named X:
- 'X_answer' : the text of the model's answer
- 'X_model'  : the name of the model that produced it

This allows side-by-side comparison of multiple answer strategies per question.
Example: "zeroshot_answer", "zeroshot_model", "gold_answer", "gold_model".

Configuration Example (part of your config["pipeline"]["answer_generation"]):
-----------------------------------------------------------------------------
answer_generation:
  run: true
  local_dataset_path: data/example/answered_questions
  # Each strategy block includes:
  #  - name (str): used to name output columns (e.g., "zeroshot_answer", "zeroshot_model")
  #  - prompt (str): which prompt from yourbench.utils.prompts to fill
  #  - model_name (str): which model key from model_roles to use
  #  - (optionally) any extra instructions or formatting
  strategies:
    - name: "zeroshot"
      prompt: "ZEROSHOT_QA_USER_PROMPT"
      model_name: "deepseek-ai/DeepSeek-V3"

    - name: "gold"
      prompt: "GOLD_QA_USER_PROMPT"
      model_name: "deepseek-ai/DeepSeek-V3"

Usage Steps:
------------
1) Load the question-level dataset (which must have a "question" column).
2) For each row (one question), for each strategy:
   - Build the prompt from the config-specified template.
   - Perform inference with the designated model.
   - Extract the <answer> block from the model's response (or store "" if not found).
3) Add new columns {strategy}_answer and {strategy}_model to store these results.
4) Save final dataset (optionally push to HF Hub).

Notes on Implementation:
------------------------
- We re-use 'run_inference' from inference_engine.py to handle concurrency, retries, etc.
- We parse <answer> content from the raw string using a small helper function.
- We log warnings if the <answer> block is missing or JSON fails, so you can diagnose issues.
- The approach is easily extended to more strategies (RAG, etc.) by editing the config.
"""

import json
import re
from typing import Dict, Any, List
from loguru import logger
from datasets import Dataset

from yourbench.utils.dataset_engine import custom_load_dataset
from yourbench.utils.dataset_engine import custom_save_dataset
from yourbench.utils.inference_engine import InferenceCall, run_inference
from yourbench.utils.prompts import (
    ZEROSHOT_QA_USER_PROMPT,
    GOLD_QA_USER_PROMPT,
)
# If you have additional prompts for RAG etc., just import them or load them dynamically.

def run(config: Dict[str, Any]) -> None:
    """
    Main entry point for answer generation.

    Steps:
      1. Read config["pipeline"]["answer_generation"] to see if run is enabled and to get dataset names.
      2. Load the input question-level dataset from "question_dataset_name".
      3. Read the list of "strategies". For each strategy, gather which model to use and which prompt template to fill.
      4. Build an InferenceCall for each row-strategy pair.
      5. Run them all in parallel via run_inference(...).
      6. Re-map each response to the corresponding row. Extract <answer> from the raw text.
      7. Append new columns to the dataset: {strategy_name}_answer, {strategy_name}_model.
      8. Save the final dataset to disk and (optionally) push to HF Hub.

    Raises:
        ValueError: if no strategies are configured or question_dataset_name is missing.
    """
    stage_cfg = config.get("pipeline", {}).get("answer_generation", {})
    if not stage_cfg.get("run", False):
        logger.info("answer_generation stage is disabled. Skipping.")
        return

    # 1. Basic config validation
    question_type = stage_cfg.get("question_type")
    if question_type == "single_shot":
        question_step_name = "single_shot_question_generation"
    elif question_type == "multi_hop":
        question_step_name = "multi_hop_question_generation"
    else:
        logger.error(f"Unsuported question_type given: {question_type}")

    strategies = stage_cfg.get("strategies", [])
    if not strategies:
        logger.critical("No strategies defined in answer_generation config. Exiting.")
        return
    
    question_dataset = custom_load_dataset(config=config, step_name=question_step_name)

    # 2. Build InferenceCalls for each row-strategy pair
    all_inference_calls: List[InferenceCall] = []
    call_index_to_row_strat: List[tuple] = []

    # We'll store the prompts in a dict for quick reference
    # by default, load from the known prompt variables (e.g., ZEROSHOT_QA_USER_PROMPT).
    # you could adapt to do `getattr(yourbench.utils.prompts, strategy["prompt"])`
    # if you have arbitrary prompt names.
    known_prompts = {
        "ZEROSHOT_QA_USER_PROMPT": ZEROSHOT_QA_USER_PROMPT,
        "GOLD_QA_USER_PROMPT": GOLD_QA_USER_PROMPT,
    }

    for row_idx, row in enumerate(question_dataset):
        # We'll expect "question" but let's handle if it's missing
        question_text = row.get("question", "").strip()
        if not question_text:
            logger.debug("Row {} has no question text. We'll produce empty answers for that row.", row_idx)

        # Optionally, we can fetch 'self_answer' or 'document_summary' from row if needed
        # for the GOLD prompt. For safety, default to blank if missing.
        doc_summary = row.get("document_summary", "")
        chunk_text = row.get("chunks", [""])[0]  # in case you want it. Not strictly needed.

        for strat in strategies:
            strat_name = strat["name"]
            prompt_key = strat["prompt"]  # e.g. "ZEROSHOT_QA_USER_PROMPT"
            model_name = strat["model_name"]
            # Load the template
            prompt_template = known_prompts.get(prompt_key, "")
            if not prompt_template:
                logger.warning(
                    "No known prompt template found for key='%s'. Using empty prompt.",
                    prompt_key
                )
                prompt_template = ""

            # Fill the user prompt
            # For 'gold' we might use doc_summary, or the chunk text, etc. 
            # This depends on how your prompt is structured. We'll keep it minimal here.
            if "GOLD_QA_USER_PROMPT" in prompt_key:
                user_prompt = prompt_template.format(
                    question=question_text,
                    summary=doc_summary,
                    document=chunk_text,
                )
            else:
                # zero-shot
                user_prompt = prompt_template.format(question=question_text)

            user_message = {"role": "user", "content": user_prompt}
            inference_call = InferenceCall(
                messages=[user_message],  # no system message needed, but you can add if desired
                tags=[f"answer_generation_{strat_name}"]
            )
            all_inference_calls.append(inference_call)
            call_index_to_row_strat.append((row_idx, strat_name, model_name))

    if not all_inference_calls:
        logger.warning("No inference calls built. Possibly empty dataset. Exiting.")
        return

    # 3. Actually run the inference using the model specified for each strategy
    #    We must group calls by their model_name, so each model can be concurrency-limited.
    #    However, the code in inference_engine expects us to pass calls grouped by step_name,
    #    and it figures out the relevant models from config["model_roles"][step_name].
    #    So we must do something special:
    #       a) In config["model_roles"]["answer_generation"], put all possible model_name used by strategies.
    #       b) We'll pass them all at once to 'run_inference'. Then we index into the response dictionary.
    #
    #    But we want each call to be routed to the correct model. The "run_inference" approach typically
    #    calls the same model for all calls. So we do a small hack: we run them in separate "batches" by model,
    #    or we patch the code to do a custom approach. 
    #
    # For simplicity (and to keep code consistent with single_shot_question_generation), we do a "manual" approach:
    #   - Group calls by model_name
    #   - For each group, run them in parallel. Then store results in a big array in the same order. 
    #   - We'll unify them after. 
    # This is simpler than rewriting the entire inference_engine.

    # Let's gather all distinct model_names from strategies
    distinct_model_names = list({s["model_name"] for s in strategies})
    # We'll build a mapping from model_name -> list of calls
    model_name_to_calls = {m: [] for m in distinct_model_names}
    model_name_to_indexes = {m: [] for m in distinct_model_names}

    for i, (row_idx, strat_name, model_name) in enumerate(call_index_to_row_strat):
        model_name_to_calls[model_name].append(all_inference_calls[i])
        model_name_to_indexes[model_name].append(i)

    # We'll create a final placeholder for each call's raw response
    raw_responses = ["" for _ in range(len(all_inference_calls))]

    # We'll loop over each distinct model, run inference in a chunk, and put the results in the correct slots
    for model_nm in distinct_model_names:
        # We reuse 'run_inference' but we must trick it into picking only the one model.
        # That means we must ensure config["model_roles"]["answer_generation"] = [model_nm], run them, revert, etc.
        # Or we can do a quick patch approach:
        original_roles = config["model_roles"].get("answer_generation", [])
        config["model_roles"]["answer_generation"] = [model_nm]

        calls_for_model = model_name_to_calls[model_nm]
        logger.info("Running inference for model '{}', total calls = {}", model_nm, len(calls_for_model))
        if not calls_for_model:
            continue

        # We get a dictionary {model_nm: [resp_1, resp_2, ...]} from run_inference
        responses_dict = run_inference(
            config=config,
            step_name="answer_generation",
            inference_calls=calls_for_model
        )
        # restore roles
        config["model_roles"]["answer_generation"] = original_roles

        # Pull the array of raw responses for that model
        model_resps = responses_dict.get(model_nm, [])
        if len(model_resps) != len(calls_for_model):
            logger.error(
                "Model '{}' returned {} responses but we expected {}. Some calls missing.",
                model_nm, len(model_resps), len(calls_for_model)
            )

        # Place them into the final raw_responses in correct order
        for local_idx, r_txt in enumerate(model_resps):
            global_idx = model_name_to_indexes[model_nm][local_idx]
            raw_responses[global_idx] = r_txt

    # 4. Now we parse out <answer> from raw_responses, for each row-strategy
    #    We'll build a structure to store the final answers in memory. 
    #    For each strategy name, we store a dict row_idx -> answer
    #    Similarly row_idx -> model
    strategy_to_answers = {}
    strategy_to_model = {}

    for i, response_text in enumerate(raw_responses):
        (row_idx, strat_name, model_name) = call_index_to_row_strat[i]
        extracted_ans = _extract_answer(response_text)
        if strat_name not in strategy_to_answers:
            strategy_to_answers[strat_name] = {}
            strategy_to_model[strat_name] = {}
        strategy_to_answers[strat_name][row_idx] = extracted_ans
        strategy_to_model[strat_name][row_idx] = model_name

    # 5. Add new columns for each strategy: "strat_name_answer", "strat_name_model".
    #    We'll do so by constructing arrays in dataset order (row_idx ascending).
    def build_col_for_strategy(strat_n: str, is_answer=True):
        col_data = []
        if is_answer:
            for r_i in range(len(question_dataset)):
                col_data.append(strategy_to_answers[strat_n].get(r_i, ""))
        else:
            for r_i in range(len(question_dataset)):
                col_data.append(strategy_to_model[strat_n].get(r_i, ""))
        return col_data

    for strat in strategies:
        sn = strat["name"]  # e.g. "zeroshot"
        answer_col_name = f"{sn}_answer"
        model_col_name = f"{sn}_model"
        logger.info("Adding columns '{}' and '{}' to dataset for strategy='{}'.", answer_col_name, model_col_name, sn)
        question_dataset = question_dataset.add_column(answer_col_name, build_col_for_strategy(sn, is_answer=True))
        question_dataset = question_dataset.add_column(model_col_name, build_col_for_strategy(sn, is_answer=False))

    # 6. Save the dataset
    logger.info("Saving answered dataset")
    custom_save_dataset(
        dataset=question_dataset,
        step_name="answer_generation",
        config=config,
    )
    logger.success("Answer generation stage completed successfully.")


def _extract_answer(raw_response: str) -> str:
    """
    Extract the text enclosed in <answer>...</answer>. 
    Returns an empty string if not found or if raw_response is empty.
    """
    if not raw_response:
        return ""
    pattern = r"<answer\s*>([\s\S]*?)</answer>"
    match = re.search(pattern, raw_response)
    if match:
        return match.group(1).strip()
    else:
        logger.warning("No <answer> tag found in response. Returning empty string.")
        return ""
