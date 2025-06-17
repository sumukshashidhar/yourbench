"""
Question Generation Pipeline (Single-Hop, Multi-Hop & Cross-Document)

This module defines a pipeline for generating question-answer pairs using either
single document chunks (single-hop), multiple chunks (multi-hop), or
chunks from multiple documents (cross-document). It supports
prompt-based inference via a language model, parses responses, and saves the output.

Features:
- Configurable chunk sampling (by count or percentage)
- Prompt formatting for single-hop and multi-hop generation
- Cross-document questions using existing multi-hop infrastructure
- Response parsing and validation
- Integration with HuggingFace Datasets and custom I/O

Main Functions:
- run_single_shot(): Generates single-hop questions.
- run_multi_hop(): Generates multi-hop questions.
- run_cross_document(): Generates cross-document questions.
"""

from __future__ import annotations
from typing import Any

from loguru import logger

from datasets import Dataset
from yourbench.utils.prompts import (
    QUESTION_GENERATION_SYSTEM_PROMPT,
    QUESTION_GENERATION_SYSTEM_PROMPT_MULTI,
    MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT,
    MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT_MULTI,
)
from yourbench.utils.chunking_utils import get_sampling_cfg
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset, _create_cross_document_dataset
from yourbench.utils.parsing_engine import (
    parse_multi_hop_responses,
    parse_single_shot_responses,
)
from yourbench.utils.inference.inference_core import run_inference
from yourbench.utils.inference.inference_builders import (
    build_multi_hop_inference_calls,
    build_single_shot_inference_calls,
)


SINGLE_SHOT_KEY = "single_shot_question_generation"
MULTI_HOP_KEY = "multi_hop_question_generation"
CROSS_DOCUMENT_KEY = "cross_document_question_generation"


def run_single_shot(config: dict[str, Any]) -> None:
    """
    Orchestrates the single-hop question generation pipeline.
    """
    stage_cfg = config.get("pipeline", {}).get(SINGLE_SHOT_KEY, {})
    if not stage_cfg.get("run", False):
        logger.info("single_shot_question_generation stage is disabled.")
        return

    question_mode = stage_cfg.get("question_mode", "open-ended")
    allowed_types = {"open-ended", "multi-choice"}
    if question_mode not in allowed_types:
        logger.warning(f"Invalid question_mode '{question_mode}', defaulting to 'open-ended'")
        question_mode = "open-ended"

    logger.info(f"Single-shot question_mode: {question_mode}")

    if question_mode == "multi-choice":
        system_prompt = QUESTION_GENERATION_SYSTEM_PROMPT_MULTI
        logger.debug("Using MULTI-CHOICE prompt for single-shot generation.")
    else:
        system_prompt = QUESTION_GENERATION_SYSTEM_PROMPT
        logger.debug("Using OPEN-ENDED prompt for single-shot generation.")

    system_msg = {"role": "system", "content": system_prompt}

    dataset = custom_load_dataset(config=config, subset="chunked")
    logger.info(f"Loaded {len(dataset)} chunks for single-shot.")

    sampling_cfg = get_sampling_cfg(stage_cfg)

    inference_calls, inference_index_map = build_single_shot_inference_calls(
        dataset, system_msg, stage_cfg, sampling_cfg
    )
    if not inference_calls:
        logger.warning("No valid inference calls for single-shot.")
        return

    responses = run_inference(config=config, step_name=SINGLE_SHOT_KEY, inference_calls=inference_calls)
    final_rows = parse_single_shot_responses(responses, inference_index_map, stage_cfg)

    if final_rows:
        logger.info(f"Saving {len(final_rows)} single-shot questions.")
        custom_save_dataset(Dataset.from_list(final_rows), config=config, subset="single_shot_questions")


def run_multi_hop(config: dict[str, Any]) -> None:
    """
    Orchestrates both multi-hop and cross-document question generation pipelines,
    if enabled in config
    """
    stage_cfg = config.get("pipeline", {}).get(MULTI_HOP_KEY, {})
    cross_cfg = stage_cfg.get("cross_document", {})
    run_multi = stage_cfg.get("run", False)
    run_cross = cross_cfg.get("enable", False)

    if not run_multi:
        logger.info("Multi-hop question generation is disabled.")
        return

    question_mode = stage_cfg.get("question_mode", "open-ended")
    if question_mode not in {"open-ended", "multi-choice"}:
        logger.warning(f"Invalid question_mode '{question_mode}', defaulting to 'open-ended'")
        question_mode = "open-ended"

    system_prompt = (
        MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT_MULTI
        if question_mode == "multi-choice"
        else MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT
    )
    system_msg = {"role": "system", "content": system_prompt}

    chunked_ds = custom_load_dataset(config=config, subset="chunked")
    logger.info(f"Loaded {len(chunked_ds)} documents for multi-hop processing.")

    def _run_and_save(dataset, label: str):
        if not dataset or len(dataset) == 0:
            logger.warning(f"No valid {label} dataset found. Skipping.")
            return
        inference_calls, inference_index_map = build_multi_hop_inference_calls(dataset, system_msg, stage_cfg)
        if not inference_calls:
            logger.warning(f"No valid inference calls for {label}.")
            return
        responses = run_inference(config=config, step_name=MULTI_HOP_KEY, inference_calls=inference_calls)
        final_rows = parse_multi_hop_responses(responses, inference_index_map, stage_cfg)
        if final_rows:
            logger.info(f"Saving {len(final_rows)} {label} questions.")
            custom_save_dataset(Dataset.from_list(final_rows), config=config, subset=label)
        else:
            logger.info(f"No valid {label} questions parsed.")

    # Run standard multi-hop if enabled
    _run_and_save(chunked_ds, "multi_hop_questions")

    # Run cross-document if enabled
    if run_cross:
        logger.info("Starting cross-document question generation.")
        cross_ds = _create_cross_document_dataset(chunked_ds, cross_cfg)
        logger.info(f"Generated {len(cross_ds)} cross-document combinations.")
        _run_and_save(cross_ds, "cross_document_questions")
