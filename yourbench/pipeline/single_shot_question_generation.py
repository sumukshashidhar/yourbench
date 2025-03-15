# yourbench/pipeline/single_shot_question_generation.py

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
from dataclasses import dataclass, field
from typing import List, Dict, Any

from loguru import logger
from datasets import Dataset

from yourbench.utils.dataset_engine import smart_load_dataset
from yourbench.utils.dataset_engine import save_dataset
from yourbench.utils.inference_engine import InferenceCall, run_inference
from yourbench.utils.prompts import (
    QUESTION_GENERATION_SYSTEM_PROMPT,
    QUESTION_GENERATION_USER_PROMPT,
)

@dataclass
class SingleHopQuestionRow:
    """
    Minimal question row. 
    chunk_id: which single-hop chunk this question came from
    document_id: the original document
    question, self_answer, etc.: the normal fields
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
    stage_cfg = config.get("pipeline", {}).get("single_shot_question_generation", {})
    if not stage_cfg.get("run", False):
        logger.info("single_shot_question_generation stage is disabled. Skipping.")
        return

    source_dataset_name = stage_cfg["source_dataset_name"]
    output_dataset_name = stage_cfg["output_dataset_name"]
    logger.info("Loading chunked dataset: {}", source_dataset_name)
    dataset = smart_load_dataset(source_dataset_name, config)
    logger.info("Loaded dataset with {} rows.", len(dataset))

    system_msg = {"role": "system", "content": QUESTION_GENERATION_SYSTEM_PROMPT}
    all_inference_calls: List[InferenceCall] = []
    call_index_map: List[tuple] = []

    for row_idx, row in enumerate(dataset):
        doc_summary = row.get("document_summary", "No summary available.")
        title = row.get("document_filename", f"Document_{row_idx}")
        doc_id = row.get("document_id", f"doc_{row_idx}")

        # For single-hop, we rely on row["chunks"], each chunk has "chunk_id" and "chunk_text".
        single_hop_chunks = row.get("chunks", [])
        if not isinstance(single_hop_chunks, list) or not single_hop_chunks:
            continue

        for c_idx, chunk_dict in enumerate(single_hop_chunks):
            if not isinstance(chunk_dict, dict):
                # fallback if old format
                chunk_text = str(chunk_dict)
                chunk_id = f"{doc_id}_{c_idx}"
            else:
                chunk_text = chunk_dict.get("chunk_text", "")
                chunk_id = chunk_dict.get("chunk_id", f"{doc_id}_{c_idx}")

            user_prompt_str = QUESTION_GENERATION_USER_PROMPT.format(
                title=title,
                document_summary=doc_summary,
                text_chunk=chunk_text,
                additional_instructions=stage_cfg.get("additional_instructions", "undergraduate")
            )
            user_msg = {"role": "user", "content": user_prompt_str}
            inference_call = InferenceCall(
                messages=[system_msg, user_msg],
                tags=["single_shot_qa"]
            )
            all_inference_calls.append(inference_call)
            call_index_map.append((row_idx, doc_id, chunk_id))

    if not all_inference_calls:
        logger.warning("No chunks found. Exiting single_shot_question_generation.")
        return

    logger.info("Sending {} calls to inference for single-shot QG.", len(all_inference_calls))
    responses_dict = run_inference(
        config=config,
        step_name="single_shot_question_generation",
        inference_calls=all_inference_calls,
    )

    question_dataset_rows: List[Dict[str, Any]] = []

    for model_name, model_responses in responses_dict.items():
        logger.info("Processing {} responses for model: {}", len(model_responses), model_name)
        if len(model_responses) != len(call_index_map):
            logger.error(
                "Model '{}' returned {} responses but expected {}. Some mismatch or truncation.",
                model_name, len(model_responses), len(call_index_map)
            )
        for idx, raw_resp in enumerate(model_responses):
            if idx >= len(call_index_map):
                break
            row_idx, doc_id, chunk_id = call_index_map[idx]

            # Extract JSON with appropriate error handling
            try:
                json_str = _extract_output_json(raw_resp)
                if not json_str.strip():
                    logger.warning("No parseable JSON found for row_idx={}, chunk_id={} (model={}).", row_idx, chunk_id, model_name)
                    continue

                question_answer_pairs = json.loads(json_str)
            except Exception as e:
                logger.warning("JSON parse error row={} chunk={} model={}: {}", row_idx, chunk_id, model_name, e)
                continue

            # Validate the type of question_answer_pairs
            if not isinstance(question_answer_pairs, list):
                logger.warning("JSON is not a list for row={}, chunk_id={} (model={}).", row_idx, chunk_id, model_name)
                continue

            # Process each QA pair with robust error handling
            for qap in question_answer_pairs:
                try:
                    # Type checking to catch non-dictionary values
                    if not isinstance(qap, dict):
                        logger.warning("Expected dictionary but got {} in row={}, chunk={}, model={}",
                                      type(qap).__name__, row_idx, chunk_id, model_name)
                        continue
                    
                    # Extract fields with type validation
                    question = qap.get("question", "")
                    self_answer = qap.get("answer", "")
                    
                    # Handle potential non-integer difficulty values
                    difficulty_raw = qap.get("estimated_difficulty", 5)
                    try:
                        difficulty = int(difficulty_raw)
                    except (ValueError, TypeError):
                        logger.warning("Invalid difficulty value '{}' for chunk_id={}, defaulting to 5", 
                                      difficulty_raw, chunk_id)
                        difficulty = 5
                    
                    qtype = qap.get("question_type", "unknown")
                    if not isinstance(qtype, str):
                        qtype = str(qtype)
                        
                    thought_process = qap.get("thought_process", "")

                    # Create and add the question row
                    question_row = SingleHopQuestionRow(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        question=question,
                        self_answer=self_answer,
                        estimated_difficulty=difficulty,
                        self_assessed_question_type=qtype,
                        generating_model=model_name,
                        thought_process=thought_process
                    )
                    question_dataset_rows.append(question_row.__dict__)
                except Exception as e:
                    # Catch any other unexpected errors during processing
                    logger.warning("Error processing QA pair for row={}, chunk={}, model={}: {}", 
                                  row_idx, chunk_id, model_name, e)
                    continue

    if not question_dataset_rows:
        logger.warning("No valid questions produced. Exiting single_shot_question_generation.")
        return

    logger.info("Constructing question-level dataset with {} rows.", len(question_dataset_rows))
    question_dataset = Dataset.from_dict({
        k: [d[k] for d in question_dataset_rows]
        for k in question_dataset_rows[0].keys()
    })

    logger.info("Saving single-shot questions to '{}'.", output_dataset_name)
    save_dataset(
        dataset=question_dataset,
        step_name="single_shot_question_generation",
        config=config,
        output_dataset_name=output_dataset_name
    )
    logger.success("Single-hop question generation completed successfully.")


def _extract_tag_content(text: str, tag: str) -> str:
    pattern = fr"<{tag}\s*>([\s\S]*?)</{tag}>"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _extract_output_json(raw_response: str) -> str:
    extracted = _extract_tag_content(raw_response, "output_json")
    if extracted.strip():
        sanitized = _maybe_strip_triple_backticks(extracted)
        if sanitized.strip():
            return sanitized

    fence_pattern = r"```json\s*([\s\S]*?)\s*```"
    fence_match = re.search(fence_pattern, raw_response)
    if fence_match:
        return fence_match.group(1).strip()

    fallback_candidates = _best_effort_json_extract(raw_response)
    return fallback_candidates[0] if fallback_candidates else ""


def _maybe_strip_triple_backticks(text_in: str) -> str:
    pattern = r"^\s*```(?:json)?\s*([\s\S]*?)\s*```$"
    m = re.match(pattern, text_in)
    if m:
        return m.group(1)
    return text_in


def _best_effort_json_extract(full_text: str) -> List[str]:
    pattern = r"([\[{].*?[\]}])"
    matches = re.findall(pattern, full_text, flags=re.DOTALL)
    candidates = []
    for m in matches:
        if (m.startswith("[") and m.endswith("]")) or (m.startswith("{") and m.endswith("}")):
            candidates.append(m.strip())
    return candidates
