import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

from loguru import logger

from yourbench.utils.dataset_engine import smart_load_dataset
from yourbench.utils.saving_engine import save_dataset
from yourbench.utils.inference_engine import InferenceCall, run_inference
from yourbench.utils.prompts import (
    QUESTION_GENERATION_SYSTEM_PROMPT,
    QUESTION_GENERATION_USER_PROMPT,
)


@dataclass
class SingleShotQuestion:
    """
    Data structure to hold one generated question-answer pair.
    """
    chunk_id: str
    document_id: str
    question: str
    answer: str
    self_estimated_difficulty_rating: int
    self_assessed_question_type: str
    self_answer: str

def run(config: Dict[str, Any]) -> None:
    """
    Run the single-shot question generation stage of the pipeline, but accumulate
    all the calls first so that the inference engine can process them in parallel.

    Steps:
      1. Load the chunked dataset using 'source_dataset_name'.
      2. Accumulate all (row, chunk) calls as a list of InferenceCall objects.
      3. Use run_inference() once for the entire list of InferenceCall objects.
      4. Parse all the returned responses, grouping them back by row.
      5. Save the resulting dataset with "single_shot_questions" and
         "generating_model" columns appended.
    """
    # === Validate stage config ===
    stage_cfg = config.get("pipeline", {}).get("single_shot_question_generation", {})
    if not stage_cfg.get("run", False):
        logger.info("single_shot_question_generation stage is disabled. Skipping.")
        return

    # Read config fields
    source_dataset_name = stage_cfg["source_dataset_name"]
    output_dataset_name = stage_cfg["output_dataset_name"]
    test_audience = stage_cfg.get("test_audience", "undergraduate")
    use_multihop = stage_cfg.get("use_multihop", False)

    logger.info("Loading chunked dataset from: {}", source_dataset_name)
    dataset = smart_load_dataset(source_dataset_name, config)
    logger.info("Loaded dataset with {} rows.", len(dataset))

    generating_model_name = config["model_roles"]["single_shot_question_generation"][0]

    # We will accumulate a list of all inference calls across all rows & chunks
    # so we can pass them into run_inference in one shot.
    all_inference_calls = []
    # We'll keep track of the mapping from index -> (row_idx, chunk_idx)
    # so we can reconstruct the results.
    call_index_to_row_chunk = []

    # Prepare system prompt once
    system_message = {
        "role": "system",
        "content": QUESTION_GENERATION_SYSTEM_PROMPT
    }

    # === Accumulate InferenceCall objects for each row/chunk ===
    for row_idx, row in enumerate(dataset):
        relevant_chunks = row["multihop_chunks"] if use_multihop else row["chunks"]

        doc_summary = row.get("document_summary", "No summary available.")
        title = row.get("document_filename", f"Document {row_idx}")

        for c_idx, chunk_text in enumerate(relevant_chunks):
            user_message_content = QUESTION_GENERATION_USER_PROMPT.format(
                title=title,
                document_summary=doc_summary,
                text_chunk=chunk_text,
                test_audience=test_audience
            )
            user_message = {"role": "user", "content": user_message_content}

            # Construct the InferenceCall
            inference_call = InferenceCall(
                messages=[system_message, user_message],
                tags=["single_shot_qa"]
            )

            # Append to the list
            all_inference_calls.append(inference_call)
            call_index_to_row_chunk.append((row_idx, c_idx))

    logger.info("Sending {} total calls to the inference engine in a single batch.", len(all_inference_calls))

    # === Run inference once on the entire batch ===
    # This is where the concurrency/batching can happen inside `run_inference`.
    model_responses_dict = run_inference(
        config=config,
        step_name="single_shot_question_generation",
        inference_calls=all_inference_calls
    )
    # model_responses_dict will look like { model_name: [list_of_responses_in_order] }
    # Since there's only one generating_model_name in this stage, we just get that list:
    model_responses = model_responses_dict[generating_model_name]

    # Now we parse the results, grouping them back by row.
    # We'll create an empty sub-list for each row to store its generated QAs.
    all_generated_qa = [[] for _ in range(len(dataset))]

    for call_idx, raw_response in enumerate(model_responses):
        row_idx, c_idx = call_index_to_row_chunk[call_idx]
        # Extract the JSON block
        extracted_json_str = _extract_tag_content(raw_response, "output_json")

        if not extracted_json_str.strip():
            logger.warning("No <output_json> block found for row {}, chunk {}. Skipping.", row_idx, c_idx)
            continue

        try:
            question_answer_pairs = json.loads(extracted_json_str)
            # For each Q-A pair, build our SingleShotQuestion
            for qap in question_answer_pairs:
                chunk_id = f"{row_idx}_{c_idx}"
                document_id = dataset[row_idx].get("document_id", f"doc_{row_idx}")
                question = qap.get("question", "")
                answer = qap.get("answer", "")
                self_difficulty = qap.get("estimated_difficulty", 5)
                question_type = qap.get("question_type", "unknown")
                self_answer = qap.get("thought_process", "")

                single_shot_q = SingleShotQuestion(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    question=question,
                    answer=answer,
                    self_estimated_difficulty_rating=self_difficulty,
                    self_assessed_question_type=question_type,
                    self_answer=self_answer
                )
                # Append as a dict
                all_generated_qa[row_idx].append({
                    "chunk_id": single_shot_q.chunk_id,
                    "document_id": single_shot_q.document_id,
                    "question": single_shot_q.question,
                    "answer": single_shot_q.answer,
                    "self_estimated_difficulty_rating": single_shot_q.self_estimated_difficulty_rating,
                    "self_assessed_question_type": single_shot_q.self_assessed_question_type,
                    "self_answer": single_shot_q.self_answer,
                })
        except Exception as e:
            logger.warning("Failed to parse JSON for row {}, chunk {}: {}", row_idx, c_idx, e)

    # === Add the new columns and save ===
    dataset = dataset.add_column("single_shot_questions", all_generated_qa)
    dataset = dataset.add_column("generating_model", [generating_model_name] * len(dataset))

    logger.info("Saving single-shot questions dataset to {}.", output_dataset_name)
    save_dataset(
        dataset=dataset,
        step_name="single_shot_question_generation",
        config=config,
        output_dataset_name=output_dataset_name
    )
    logger.success("Single-shot question generation complete.")


def _extract_tag_content(text: str, tag: str) -> str:
    """
    Extract content enclosed in <tag> ... </tag> (non-greedy).
    Returns the first match or empty string if none found.
    """
    import re
    pattern = fr"<{tag}\s*>([\s\S]*?)</{tag}>"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return ""
