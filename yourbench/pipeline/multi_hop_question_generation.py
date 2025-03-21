# ============================================================
# multi_hop_question_generation.py
# ============================================================
"""
Author: @sumukshashidhar

Module Name:
------------
multi_hop_question_generation

Purpose:
--------
This module implements the multi-hop question generation stage within the YourBench pipeline.
It processes a dataset of documents—each containing a list of multi-hop chunks—and generates
multi-hop questions requiring integrative reasoning across those chunks. It uses a large
language model to produce question-answer pairs in JSON format.

Usage:
------
This module is typically invoked as part of the overall YourBench pipeline. It expects:
1. A source dataset (e.g., documents with 'multihop_chunks' field).
2. Configuration for multi-hop question generation, such as sampling parameters and
   additional instructions.
3. The pipeline orchestrator (in `handler.py`) calls `run(config)` if
   `multi_hop_question_generation` is enabled in the YAML configuration.

The module then:
1. Optionally samples multi-hop chunks from each document.
2. Prompts a large language model to generate multi-hop question-answer pairs.
3. Parses and saves the generated questions in a structured dataset.

Error Handling and Logging:
---------------------------
- Comprehensive logging is performed using `loguru` at various levels to trace execution.
- Exceptions are caught and logged as errors, with the module attempting to continue
  where practical.
- Critical issues produce warnings or errors and gracefully terminate the stage.

Module-Level Dependencies:
--------------------------
- Relies on the shared pipeline utilities (e.g., `yourbench.utils.dataset_engine`,
  `yourbench.utils.inference_engine`, `yourbench.utils.prompts`).
- Preserves the existing signature and functionality for downstream consistency.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from datasets import Dataset
from loguru import logger

from yourbench.utils.dataset_engine import (
    custom_load_dataset,
    custom_save_dataset,
)
from yourbench.utils.inference_engine import InferenceCall, run_inference
from yourbench.utils.prompts import (
    MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT,
    MULTI_HOP_QUESTION_GENERATION_USER_PROMPT,
)

# Import the unified parsing function
from yourbench.utils.parsing_engine import parse_qa_pairs_from_response


@dataclass
class QuestionAnswerPair:
    """
    Data structure to represent a question-answer pair returned by the model.
    """

    question: str
    answer: str
    estimated_difficulty: int = 5
    question_type: str = "unknown"
    thought_process: str = ""
    citations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Normalize fields
        self.question = str(self.question).strip()
        self.answer = str(self.answer).strip()
        self.estimated_difficulty = _force_int_in_range(
            self.estimated_difficulty, 1, 10
        )
        self.question_type = str(self.question_type)
        self.thought_process = str(self.thought_process)
        if not isinstance(self.citations, list):
            self.citations = []


@dataclass
class MultiHopQuestionRow:
    """
    Data structure to represent a single multi-hop question row.
    """

    document_id: str
    source_chunk_ids: List[str]
    question: str
    self_answer: str
    estimated_difficulty: int
    self_assessed_question_type: str
    generating_model: str
    thought_process: str
    citations: List[str] = field(default_factory=list)
    raw_response: str = field(default="")

    @classmethod
    def from_qa_pair(
        cls,
        qa_pair: QuestionAnswerPair,
        document_id: str,
        source_chunk_ids: List[str],
        generating_model: str,
        raw_response: str = "",
    ) -> "MultiHopQuestionRow":
        return cls(
            document_id=document_id,
            source_chunk_ids=source_chunk_ids,
            question=qa_pair.question,
            self_answer=qa_pair.answer,
            estimated_difficulty=qa_pair.estimated_difficulty,
            self_assessed_question_type=qa_pair.question_type,
            generating_model=generating_model,
            thought_process=qa_pair.thought_process,
            citations=qa_pair.citations,
            raw_response=raw_response,
        )


def run(config: Dict[str, Any]) -> None:
    """
    Execute the multi-hop question generation stage.
    """
    stage_cfg = config.get("pipeline", {}).get("multi_hop_question_generation", {})
    if not stage_cfg.get("run", False):
        logger.info("multi_hop_question_generation stage is disabled. Skipping.")
        return

    # 1) Dataset Loading
    dataset = custom_load_dataset(config=config, subset="chunked")
    logger.info(f"Loaded chunked subset with {len(dataset)} rows for Multi-hop question generation.")

    # 2) Build Inference Calls (including sampling)
    inference_calls, call_index_map = _multihop_chunk_sampling_and_calls(
        dataset, stage_cfg
    )

    # 3) Run Inference
    if not inference_calls:
        logger.warning("No multi-hop inference calls were created. Exiting stage.")
        return
    responses_dict = _multihop_qa_generation(config, inference_calls)

    # 4) Parse and Build Final Dataset
    final_dataset = _parse_and_build_final(config, responses_dict, call_index_map)
    if final_dataset is None or len(final_dataset) == 0:
        logger.warning("No valid multi-hop question rows produced. Exiting stage.")
        return

    # 5) Save the result
    custom_save_dataset(dataset=final_dataset, config=config, subset="multi_hop_questions")
    logger.success("Multi-hop question generation completed successfully.")


def _multihop_chunk_sampling_and_calls(dataset, stage_cfg: Dict[str, Any]):
    """
    Sample multi-hop chunks and build InferenceCalls.
    Returns:
      - inference_calls: list of InferenceCall
      - call_index_map: parallel list of (row_idx, doc_id, source_chunk_ids)
    """
    system_msg = {
        "role": "system",
        "content": MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT,
    }
    all_inference_calls = []
    call_index_map = []

    for row_idx, row in enumerate(dataset):
        doc_summary = row.get("document_summary", "No summary provided.")
        title = row.get("document_filename", f"Document_{row_idx}")
        doc_id = row.get("document_id", f"doc_{row_idx}")

        multi_hop_chunks = row.get("multihop_chunks", [])
        if not isinstance(multi_hop_chunks, list) or not multi_hop_chunks:
            logger.debug(
                f"No multi-hop chunks found in row index={row_idx}, doc_id={doc_id}. Skipping row."
            )
            continue

        chosen_multi_hops = _sample_multi_hop_chunks(
            multi_hop_chunks, stage_cfg.get("chunk_sampling", {})
        )
        if not chosen_multi_hops:
            logger.debug(
                f"Row idx={row_idx} doc_id={doc_id} had multi-hop chunks but none after sampling."
            )
            continue

        additional_instructions = stage_cfg.get(
            "additional_instructions", "undergraduate"
        )

        for mh_idx, mh_dict in enumerate(chosen_multi_hops):
            if not isinstance(mh_dict, dict):
                continue

            subchunk_ids = mh_dict.get("chunk_ids", [])
            subchunk_texts = mh_dict.get("chunks_text", [])
            if not subchunk_texts:
                logger.debug(
                    f"Empty multi-hop chunk at row_idx={row_idx}, doc_id={doc_id}. Skipping."
                )
                continue

            # Build user prompt by enumerating each subchunk
            text_chunks_aggregated = ""
            for i, sc_text in enumerate(subchunk_texts):
                text_chunks_aggregated += (
                    f"<text_chunk_{i}>{sc_text}</text_chunk_{i}>\n"
                )

            user_prompt_str = MULTI_HOP_QUESTION_GENERATION_USER_PROMPT.format(
                title=title,
                document_summary=doc_summary,
                chunks=text_chunks_aggregated,
                additional_instructions=additional_instructions,
            )
            user_msg = {"role": "user", "content": user_prompt_str}

            inference_call = InferenceCall(
                messages=[system_msg, user_msg], tags=["multi_hop_qa"]
            )
            all_inference_calls.append(inference_call)
            call_index_map.append((row_idx, doc_id, subchunk_ids))

    return all_inference_calls, call_index_map


def _sample_multi_hop_chunks(
    mh_chunks: List[Dict[str, Any]], chunk_sampling_cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Sample multi-hop chunks based on the stage configuration.
    """
    if len(chunk_sampling_cfg) == 0:
        # If there's no config, return all
        return mh_chunks

    mode = chunk_sampling_cfg.get("mode", "all").lower()
    value = chunk_sampling_cfg.get("value", 1.0)
    rand_seed = chunk_sampling_cfg.get("random_seed", 42)
    random.seed(rand_seed)

    total_multi_hops = len(mh_chunks)
    if total_multi_hops == 0:
        return mh_chunks

    if mode == "percentage":
        k = int(total_multi_hops * float(value))
        k = max(0, min(k, total_multi_hops))
        if k < total_multi_hops:
            return random.sample(mh_chunks, k)
        return mh_chunks

    elif mode == "count":
        k = min(int(value), total_multi_hops)
        if k < total_multi_hops:
            return random.sample(mh_chunks, k)
        return mh_chunks

    # Otherwise return all
    return mh_chunks


def _multihop_qa_generation(
    config: Dict[str, Any], inference_calls: List[InferenceCall]
):
    """
    Call the inference engine to get multi-hop Q&A responses.
    """
    logger.info(f"Sending {len(inference_calls)} multi-hop calls to inference...")
    return run_inference(
        config=config,
        step_name="multi_hop_question_generation",
        inference_calls=inference_calls,
    )


def _parse_and_build_final(
    config: Dict[str, Any],
    responses_dict: Dict[str, List[str]],
    call_index_map: List[tuple],
) -> Dataset:
    """
    Parse each model's responses into MultiHopQuestionRow items, then build a final dataset.
    """
    final_multi_hop_questions = []

    for model_name, model_responses in responses_dict.items():
        logger.info(
            f"Processing {len(model_responses)} responses for model: {model_name}"
        )
        if len(model_responses) != len(call_index_map):
            logger.error(
                f"Model '{model_name}' returned {len(model_responses)} responses; expected {len(call_index_map)}. Mismatch."
            )

        for idx, raw_resp in enumerate(model_responses):
            if idx >= len(call_index_map):
                break

            row_idx, doc_id, source_chunk_ids = call_index_map[idx]
            qa_pairs = parse_qa_pairs_from_response(raw_resp)

            if not qa_pairs:
                # Fallback row if no parseable QAs
                logger.warning(
                    f"No parseable JSON for row={row_idx}, doc_id={doc_id} (model={model_name}). Creating fallback row."
                )
                fallback_row = MultiHopQuestionRow(
                    document_id=doc_id,
                    source_chunk_ids=source_chunk_ids,
                    question="No question parsed",
                    self_answer="No answer parsed",
                    estimated_difficulty=5,
                    self_assessed_question_type="unknown",
                    generating_model=model_name,
                    thought_process="",
                    citations=[],
                    raw_response=raw_resp,
                )
                final_multi_hop_questions.append(fallback_row.__dict__)
                continue

            # Otherwise, process each QA pair
            for qap_dict in qa_pairs:
                try:
                    # Convert dictionary -> QuestionAnswerPair
                    pair_obj = QuestionAnswerPair(
                        question=qap_dict.get("question", ""),
                        answer=qap_dict.get("answer", ""),
                        estimated_difficulty=qap_dict.get("estimated_difficulty", 5),
                        question_type=qap_dict.get("question_type", "unknown"),
                        thought_process=qap_dict.get("thought_process", ""),
                        citations=qap_dict.get("citations", []),
                    )
                    if not pair_obj.question:
                        logger.debug(
                            f"Empty question found for row={row_idx}, doc_id={doc_id}, skipping pair."
                        )
                        continue

                    row_obj = MultiHopQuestionRow.from_qa_pair(
                        qa_pair=pair_obj,
                        document_id=doc_id,
                        source_chunk_ids=source_chunk_ids,
                        generating_model=model_name,
                        raw_response=raw_resp,
                    )
                    final_multi_hop_questions.append(row_obj.__dict__)

                except Exception as pair_error:
                    logger.warning(
                        f"Error processing QA pair for doc_id={doc_id}, skipping pair: {pair_error}"
                    )
                    continue

    if not final_multi_hop_questions:
        return None

    logger.info(
        f"Constructing multi-hop question dataset with {len(final_multi_hop_questions)} rows..."
    )
    try:
        col_keys = list(final_multi_hop_questions[0].keys())
        dataset_dict = {
            k: [row[k] for row in final_multi_hop_questions] for k in col_keys
        }
        return Dataset.from_dict(dataset_dict)
    except Exception as ds_error:
        logger.error(
            f"Failed to create dataset from multi-hop question rows: {ds_error}"
        )
        return None


def _force_int_in_range(value: Any, min_val: int, max_val: int) -> int:
    """
    Convert a value to int and clamp it between min_val and max_val.
    """
    try:
        ivalue = int(value)
    except (ValueError, TypeError):
        ivalue = (min_val + max_val) // 2
    return max(min_val, min(ivalue, max_val))
