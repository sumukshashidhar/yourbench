# ============================================================
# single_shot_question_generation.py
# ============================================================
"""
Single-Hop Question Generation Module

Minimal approach:
----------------
- Each "chunks" entry is now just:
    { "chunk_id": "...", "chunk_text": "..." }
- We only store that chunk_id in the final question row (plus document_id),
  so you can map which chunk is responsible for each question.
- We do NOT store chunk_location_id or chunk_uuid. 
- The final question dataset includes columns:
    [chunk_id, document_id, question, self_answer, estimated_difficulty, ...]
"""

import json
import re
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any

from loguru import logger
from datasets import Dataset

from yourbench.utils.dataset_engine import (
    smart_load_dataset,
    smart_get_source_dataset_name,
    smart_get_output_dataset_name,
    smart_get_source_subset,
    smart_get_output_subset,
    save_dataset
)
from yourbench.utils.inference_engine import InferenceCall, run_inference
from yourbench.utils.prompts import (
    QUESTION_GENERATION_SYSTEM_PROMPT,
    QUESTION_GENERATION_USER_PROMPT,
)


# === Data Model for storing each question row ===
@dataclass
class SingleHopQuestionRow:
    """
    Minimal question row.

    chunk_id: which single-hop chunk this question came from
    document_id: the original document
    question: the generated question text
    self_answer: a plausible answer or reasoning from the model
    estimated_difficulty: an integer from 1-10
    self_assessed_question_type: a descriptor for question style or category
    generating_model: which model produced this question
    thought_process: a free-form description of how the question was derived
    """
    chunk_id: str
    document_id: str
    question: str
    self_answer: str
    estimated_difficulty: int
    self_assessed_question_type: str
    generating_model: str
    thought_process: str


def run(config: Dict[str, Any]) -> None:
    """
    Main entry point for the single-shot question generation pipeline stage.
    
    This function:

    1. Loads the dataset containing document summaries and single-hop chunks.
    2. Optionally samples a subset of chunks according to user-defined config 
       to limit cost (fixed count or fixed percentage).
    3. For each sampled chunk, calls an LLM to generate question-answer pairs.
    4. Constructs a final question-level dataset with relevant fields.
    5. Saves the question dataset to disk or the Hugging Face Hub.

    The YAML config should contain a block like:

    single_shot_question_generation:
      run: true
      source_subset: chunked_documents
      output_subset: single_shot_questions
      additional_instructions: "Generate moderately challenging questions"
      chunk_sampling:
        mode: "percentage"  # or "count"
        value: 0.5          # if percentage, 0.5 => 50% of chunks
                            # if count, e.g., 10 => sample 10 chunks
        random_seed: 42

    The key "chunk_sampling" is optional. If not present, all chunks will be used.
    """
    stage_cfg = config.get("pipeline", {}).get("single_shot_question_generation", {})
    if not stage_cfg.get("run", False):
        logger.info("single_shot_question_generation stage is disabled. Skipping.")
        return

    # === Identify source & output dataset info from config ===
    source_dataset_name = smart_get_source_dataset_name("single_shot_question_generation", config)
    output_dataset_name = smart_get_output_dataset_name("single_shot_question_generation", config)
    source_subset = smart_get_source_subset("single_shot_question_generation", config)
    output_subset = smart_get_output_subset("single_shot_question_generation", config)

    logger.info("Loading chunked dataset for single-shot QG: {}", source_dataset_name)
    dataset = smart_load_dataset(source_dataset_name, config, dataset_subset=source_subset)
    logger.info("Loaded dataset with {} rows.", len(dataset))

    # === Prepare system prompt for question generation ===
    system_msg = {"role": "system", "content": QUESTION_GENERATION_SYSTEM_PROMPT}

    # Will collect all calls to inference and a map back to row
    all_inference_calls: List[InferenceCall] = []
    call_index_map: List[tuple] = []

    # === Helper to sample chunks according to config ===
    def sample_chunks_if_needed(chunks_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Given the config's `chunk_sampling` block, either sample a fraction or 
        a fixed number of chunks, or return them all if none specified.
        """
        chunk_sampling_cfg = stage_cfg.get("chunk_sampling", {})
        if not chunk_sampling_cfg:
            # If no sampling config is provided, return all
            return chunks_list

        mode = chunk_sampling_cfg.get("mode", "all").lower()  # 'percentage' or 'count'
        value = chunk_sampling_cfg.get("value", 1.0)
        rand_seed = chunk_sampling_cfg.get("random_seed", 42)
        random.seed(rand_seed)

        total_chunks = len(chunks_list)
        if total_chunks == 0:
            return chunks_list

        if mode == "percentage":
            # e.g. value=0.5 means 50%
            k = int(total_chunks * float(value))
            k = max(0, min(k, total_chunks))
            if k < total_chunks:
                return random.sample(chunks_list, k)
            return chunks_list

        elif mode == "count":
            # e.g. value=10 means sample 10 chunks
            k = min(int(value), total_chunks)
            if k < total_chunks:
                return random.sample(chunks_list, k)
            return chunks_list

        # Fallback if mode not recognized
        return chunks_list

    # === Loop over each row in the dataset, and sample chunks as needed ===
    for row_idx, row in enumerate(dataset):
        doc_summary = row.get("document_summary", "No summary available.")
        title = row.get("document_filename", f"Document_{row_idx}")
        doc_id = row.get("document_id", f"doc_{row_idx}")

        single_hop_chunks = row.get("chunks", [])
        if not isinstance(single_hop_chunks, list) or not single_hop_chunks:
            logger.debug("No chunks found in row index={} for doc_id={}. Skipping row.", row_idx, doc_id)
            continue

        # Sample chunks to control cost
        chosen_chunks = sample_chunks_if_needed(single_hop_chunks)

        additional_instructions = stage_cfg.get("additional_instructions", "undergraduate")

        # For each chunk, build an inference call
        for c_idx, chunk_dict in enumerate(chosen_chunks):
            if not isinstance(chunk_dict, dict):
                # Fallback if old format
                chunk_text = str(chunk_dict)
                chunk_id = f"{doc_id}_{c_idx}"
            else:
                chunk_text = chunk_dict.get("chunk_text", "")
                chunk_id = chunk_dict.get("chunk_id", f"{doc_id}_{c_idx}")

            user_prompt_str = QUESTION_GENERATION_USER_PROMPT.format(
                title=title,
                document_summary=doc_summary,
                text_chunk=chunk_text,
                additional_instructions=additional_instructions
            )
            user_msg = {"role": "user", "content": user_prompt_str}
            inference_call = InferenceCall(
                messages=[system_msg, user_msg],
                tags=["single_shot_qa"]
            )
            all_inference_calls.append(inference_call)
            call_index_map.append((row_idx, doc_id, chunk_id))

    if not all_inference_calls:
        logger.warning("No calls were created for single_shot_question_generation. Exiting.")
        return

    logger.info("Sending {} calls to inference for single-shot question generation.", len(all_inference_calls))
    # === Run the calls in parallel using the pipeline's inference engine ===
    responses_dict = run_inference(
        config=config,
        step_name="single_shot_question_generation",
        inference_calls=all_inference_calls,
    )

    # We'll store the final question dataset rows in memory
    question_dataset_rows: List[Dict[str, Any]] = []

    # For each model that responded, we have a list of responses in the same order
    for model_name, model_responses in responses_dict.items():
        logger.info("Processing {} responses from model: {}", len(model_responses), model_name)

        # Check for mismatch in number of responses
        if len(model_responses) != len(call_index_map):
            logger.error(
                "Model '{}' returned {} responses but we have {} calls. Possible mismatch.",
                model_name, len(model_responses), len(call_index_map)
            )

        # Process each raw response in order
        for idx, raw_resp in enumerate(model_responses):
            if idx >= len(call_index_map):
                break
            row_idx, doc_id, chunk_id = call_index_map[idx]

            # Extract JSON containing question-answer pairs
            json_str = _extract_output_json(raw_resp)
            if not json_str.strip():
                logger.warning(
                    "No parseable JSON found for row_idx={}, chunk_id={}, model={}. Skipping.",
                    row_idx, chunk_id, model_name
                )
                continue

            try:
                question_answer_pairs = json.loads(json_str)
            except Exception as e:
                logger.warning(
                    "JSON parse error row_idx={}, chunk_id={}, model={}: {}",
                    row_idx, chunk_id, model_name, e
                )
                continue

            if not isinstance(question_answer_pairs, list):
                logger.warning(
                    "Expected a list of QA pairs, got something else for row_idx={}, chunk_id={}, model={}.",
                    row_idx, chunk_id, model_name
                )
                continue

            # Process each QA pair
            for qap in question_answer_pairs:
                if not isinstance(qap, dict):
                    logger.warning(
                        "Invalid QA pair structure for row_idx={}, chunk_id={}, model={}. Expected dict, got {}",
                        row_idx, chunk_id, model_name, type(qap).__name__
                    )
                    continue

                # Extract fields robustly
                question_text = qap.get("question", "").strip()
                if not question_text:
                    logger.debug("Empty question found, skipping. row_idx={}, chunk_id={}", row_idx, chunk_id)
                    continue
                self_answer = qap.get("answer", "").strip()

                # Difficulty
                difficulty_raw = qap.get("estimated_difficulty", 5)
                try:
                    difficulty_val = int(difficulty_raw)
                except (ValueError, TypeError):
                    logger.warning("Invalid estimated_difficulty '{}', defaulting to 5", difficulty_raw)
                    difficulty_val = 5

                # Question type
                qtype = qap.get("question_type", "unknown")
                if not isinstance(qtype, str):
                    qtype = str(qtype)

                # Thought process
                thought_process = qap.get("thought_process", "")
                if not isinstance(thought_process, str):
                    thought_process = str(thought_process)

                # Construct row object
                question_row = SingleHopQuestionRow(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    question=question_text,
                    self_answer=self_answer,
                    estimated_difficulty=difficulty_val,
                    self_assessed_question_type=qtype,
                    generating_model=model_name,
                    thought_process=thought_process
                )
                question_dataset_rows.append(question_row.__dict__)

    if not question_dataset_rows:
        logger.warning("No valid questions produced in single_shot_question_generation. Exiting.")
        return

    logger.info("Constructing a final dataset with {} single-hop questions.", len(question_dataset_rows))
    # Convert our list of dicts to an HF Dataset
    col_names = question_dataset_rows[0].keys()
    final_data = {c: [row[c] for row in question_dataset_rows] for c in col_names}
    question_dataset = Dataset.from_dict(final_data)

    # === Save the question dataset ===
    logger.info("Saving single-shot questions to dataset name '{}'.", output_dataset_name)
    save_dataset(
        dataset=question_dataset,
        step_name="single_shot_question_generation",
        config=config,
        output_dataset_name=output_dataset_name,
        output_subset=output_subset
    )
    logger.success("Single-hop question generation completed successfully.")


# === Helper Functions ===

def _extract_tag_content(text: str, tag: str) -> str:
    """
    Extract the text enclosed in <tag>...</tag>.
    Returns empty string if not found.
    """
    pattern = fr"<{tag}\s*>([\s\S]*?)</{tag}>"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _extract_output_json(raw_response: str) -> str:
    """
    Attempt to extract JSON from the model response by searching
    for <output_json> tags or fenced code blocks with 'json'.
    """
    # 1) Check <output_json> block
    extracted = _extract_tag_content(raw_response, "output_json")
    if extracted.strip():
        sanitized = _maybe_strip_triple_backticks(extracted)
        if sanitized.strip():
            return sanitized

    # 2) Check ```json fenced block
    fence_pattern = r"```json\s*([\s\S]*?)\s*```"
    fence_match = re.search(fence_pattern, raw_response)
    if fence_match:
        return fence_match.group(1).strip()

    # 3) Attempt a best-effort bracket-based extraction
    fallback_candidates = _best_effort_json_extract(raw_response)
    return fallback_candidates[0] if fallback_candidates else ""


def _maybe_strip_triple_backticks(text_in: str) -> str:
    """
    If the text is wrapped in triple backticks (```), remove them.
    """
    pattern = r"^\s*```(?:json)?\s*([\s\S]*?)\s*```$"
    m = re.match(pattern, text_in)
    if m:
        return m.group(1)
    return text_in


def _best_effort_json_extract(full_text: str) -> List[str]:
    """
    Search for bracket-delimited content that might be valid JSON.
    Returns a list of candidate strings.
    """
    pattern = r"([\[{].*?[\]}])"
    matches = re.findall(pattern, full_text, flags=re.DOTALL)
    candidates = []
    for m in matches:
        if (m.startswith("[") and m.endswith("]")) or (m.startswith("{") and m.endswith("}")):
            candidates.append(m.strip())
    return candidates
