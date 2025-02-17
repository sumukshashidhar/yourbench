# =====================================
# single_shot_question_generation.py
# =====================================
"""
Single-Hop Question Generation Module

Refactored to:
1) Support multiple generating models in one run.
2) Produce a question-focused dataset with one row per generated question.
3) Retain the option to concatenate multiple runs if desired.

Configuration Example (from config["pipeline"]["single_shot_question_generation"]):
-----------------------------------------------------------------------------------
single_shot_question_generation:
  source_chunked_dataset_name: yb_demo_chunked_documents
  output_question_dataset_name: yb_demo_single_shot_questions
  local_dataset_path: data/example/single_shot_questions
  diversification_seed: "24 year old adult"
  concat_existing_dataset: true
  run: true

Dependencies/Assumptions:
- The pipeline config includes a "model_roles" entry for "single_shot_question_generation"
  listing all generating models you wish to use.
- The run_inference(...) call returns a dict mapping each model_name to a list of responses,
  preserving order with the "call_index_to_row_chunk" structure.
- The question-level dataset uses the columns requested:
  chunk_id,
  document_id,
  chunk_location_id,
  diversification_seed,
  question,
  self_answer,
  estimated_difficulty,
  self_assessed_question_type,
  generating_model
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

from loguru import logger
from datasets import Dataset

from yourbench.utils.dataset_engine import smart_load_dataset
from yourbench.utils.saving_engine import save_dataset
from yourbench.utils.inference_engine import InferenceCall, run_inference
from yourbench.utils.prompts import (
    QUESTION_GENERATION_SYSTEM_PROMPT,
    QUESTION_GENERATION_USER_PROMPT,
)

@dataclass
class SingleHopQuestionRow:
    """
    Data structure describing a single question as one row of the final question dataset.
    """
    chunk_id: str
    document_id: str
    chunk_location_id: int
    diversification_seed: str
    question: str
    self_answer: str
    estimated_difficulty: int
    self_assessed_question_type: str
    generating_model: str


def run(config: Dict[str, Any]) -> None:
    """
    Main entry point for single-shot question generation.

    Steps:
      1. Load chunked dataset from config["pipeline"]["single_shot_question_generation"]["source_dataset_name"].
      2. Build inference calls for each row-chunk pair. (We record row_idx, chunk_idx for reconstruction.)
      3. Call run_inference(...) to handle all calls in parallel for all specified models.
      4. Parse each model's responses into a question-level list of dictionaries:
         [
           {
             chunk_id, document_id, chunk_location_id, diversification_seed,
             question, self_answer, estimated_difficulty, self_assessed_question_type,
             generating_model
           },
           ...
         ]
      5. Convert that list into a Hugging Face Dataset, then use save_dataset(...) so that
         we can optionally concatenate with existing data and push to the hub.
    """
    stage_cfg = config.get("pipeline", {}).get("single_shot_question_generation", {})
    if not stage_cfg.get("run", False):
        logger.info("single_shot_question_generation stage is disabled. Skipping.")
        return

    # === Step 1: Load the chunked dataset ===
    source_dataset_name = stage_cfg["source_dataset_name"]
    output_dataset_name = stage_cfg["output_dataset_name"]
    diversification_seed = stage_cfg.get("diversification_seed", "generic_seed")
    use_multihop = stage_cfg.get("use_multihop", False)

    logger.info("Loading chunked dataset: {}", source_dataset_name)
    dataset = smart_load_dataset(source_dataset_name, config)
    logger.info("Loaded dataset with {} rows.", len(dataset))

    # Create the system message once. We'll reuse it for all calls.
    system_message = {"role": "system", "content": QUESTION_GENERATION_SYSTEM_PROMPT}

    # We'll accumulate all calls in a single list, across all rows and chunks.
    all_inference_calls: List[InferenceCall] = []
    call_index_to_row_chunk: List[tuple] = []

    # For each row in the dataset, for each chunk, create a prompt
    for row_idx, row in enumerate(dataset):
        # Use multi_hop or single-hop chunks
        relevant_chunks = row["multihop_chunks"] if use_multihop else row["chunks"]

        doc_summary = row.get("document_summary", "No summary available.")
        title = row.get("document_filename", f"Document_{row_idx}")

        for c_idx, chunk_text in enumerate(relevant_chunks):
            user_content = QUESTION_GENERATION_USER_PROMPT.format(
                title=title,
                document_summary=doc_summary,
                text_chunk=chunk_text,
                test_audience=stage_cfg.get("test_audience", "undergraduate")
            )
            user_message = {"role": "user", "content": user_content}
            inference_call = InferenceCall(
                messages=[system_message, user_message],
                tags=["single_shot_qa"]
            )
            all_inference_calls.append(inference_call)
            call_index_to_row_chunk.append((row_idx, c_idx))

    if not all_inference_calls:
        logger.warning("No chunks found. Exiting single_shot_question_generation.")
        return

    # === Step 2: Run inference on all calls with all models in config["model_roles"]["single_shot_question_generation"] ===
    logger.info("Sending {} total calls to inference for single-hop QG.", len(all_inference_calls))
    responses_dict = run_inference(
        config=config,
        step_name="single_shot_question_generation",
        inference_calls=all_inference_calls,
    )

    # responses_dict is { "modelA": [resp_1, resp_2, ...], "modelB": [resp_1, resp_2, ...], ... }

    # === Step 3: Parse the responses into question-level records ===
    question_dataset_rows: List[Dict[str, Any]] = []

    for model_name, model_responses in responses_dict.items():
        logger.info("Processing {} responses for model: {}", len(model_responses), model_name)

        # Ensure we have the same number of responses as calls
        if len(model_responses) != len(call_index_to_row_chunk):
            logger.error(
                "Model '{}' returned {} responses, but we expected {}. Skipping leftover items.",
                model_name, len(model_responses), len(call_index_to_row_chunk)
            )

        for call_idx, raw_response in enumerate(model_responses):
            if call_idx >= len(call_index_to_row_chunk):
                # Means the model returned extra responses somehow
                break
            row_idx, c_idx = call_index_to_row_chunk[call_idx]
            doc_id = dataset[row_idx].get("document_id", f"doc_{row_idx}")

            ### ADDED OR MODIFIED CODE HERE - parse JSON from either <output_json> or triple-backtick
            parsed_json_str = _extract_output_json(raw_response)
            if not parsed_json_str.strip():
                logger.warning(
                    "No parseable JSON found for row {}, chunk {}. Model={}",
                    row_idx, c_idx, model_name
                )
                continue

            try:
                question_answer_pairs = json.loads(parsed_json_str)
            except Exception as e:
                logger.warning(
                    "Failed to parse JSON for row {}, chunk {} (model={}): {}",
                    row_idx, c_idx, model_name, e
                )
                continue

            # Build question rows
            for qap in question_answer_pairs:
                question_text = qap.get("question", "")
                self_answer_text = qap.get("thought_process", "")
                difficulty = qap.get("estimated_difficulty", 5)
                question_type = qap.get("question_type", "unknown")

                # chunk_id can be row_{row_idx}_chunk_{c_idx}
                chunk_id = f"row_{row_idx}_chunk_{c_idx}"

                question_row = SingleHopQuestionRow(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    chunk_location_id=c_idx,
                    diversification_seed=diversification_seed,
                    question=question_text,
                    self_answer=self_answer_text,
                    estimated_difficulty=difficulty,
                    self_assessed_question_type=question_type,
                    generating_model=model_name
                )

                question_dataset_rows.append(question_row.__dict__)

    if not question_dataset_rows:
        logger.warning("No valid question rows produced. Exiting single_shot_question_generation.")
        return

    # === Step 4: Convert question_dataset_rows -> HF Dataset, and save ===
    logger.info("Constructing question-level dataset with {} rows...", len(question_dataset_rows))
    question_dataset = Dataset.from_dict({k: [d[k] for d in question_dataset_rows]
                                          for k in question_dataset_rows[0].keys()})

    logger.info("Saving question dataset to HF Hub under '{}'.", output_dataset_name)
    save_dataset(
        dataset=question_dataset,
        step_name="single_shot_question_generation",
        config=config,
        output_dataset_name=output_dataset_name
    )
    logger.success("Single-hop question generation completed successfully.")


def _extract_tag_content(text: str, tag: str) -> str:
    """
    Extract content enclosed in <tag>...</tag> (non-greedy).
    Returns the first match or empty string if none found.
    """
    pattern = fr"<{tag}\s*>([\s\S]*?)</{tag}>"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""

### ADDED OR MODIFIED CODE HERE:
def _extract_output_json(raw_response: str) -> str:
    """
    Attempt to extract JSON from either <output_json>...</output_json> or
    from a triple-backtick fenced code block: ```json ... ```.

    Returns the raw JSON string if found, otherwise empty string.
    """
    # 1. Try <output_json> first
    extracted = _extract_tag_content(raw_response, "output_json")
    if extracted.strip():
        return extracted

    # 2. If no <output_json>, look for ```json fenced content
    fence_pattern = r"```json\s*([\s\S]*?)\s*```"
    fence_match = re.search(fence_pattern, raw_response)
    if fence_match:
        return fence_match.group(1).strip()

    # 3. Nothing found
    return ""
