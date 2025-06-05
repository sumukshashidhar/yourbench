"""
Question Generation Pipeline (Single-Hop & Multi-Hop)

This module defines a pipeline for generating question-answer pairs using either
single document chunks (single-hop) or multiple chunks (multi-hop). It supports
prompt-based inference via a language model, parses responses, and saves the output.

Features:
- Configurable chunk sampling (by count or percentage)
- Prompt formatting for single-hop and multi-hop generation
- Response parsing and validation
- Integration with HuggingFace Datasets and custom I/O

Main Functions:
- run_single_shot(): Generates single-hop questions.
- run_multi_hop(): Generates multi-hop questions.
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
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
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
    Orchestrates the multi-hop question generation pipeline.
    """
    stage_cfg = config.get("pipeline", {}).get(MULTI_HOP_KEY, {})
    if not stage_cfg.get("run", False):
        logger.info("multi_hop_question_generation stage is disabled.")
        return

    question_mode = stage_cfg.get("question_mode", "open-ended")
    allowed_types = {"open-ended", "multi-choice"}
    if question_mode not in allowed_types:
        logger.warning(f"Invalid question_mode '{question_mode}', defaulting to 'open-ended'")
        question_mode = "open-ended"

    logger.info(f"Multi-hop question_mode: {question_mode}")

    if question_mode == "multi-choice":
        system_prompt = MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT_MULTI
        logger.debug("Using MULTI-CHOICE prompt for multi-hop generation.")
    else:
        system_prompt = MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT
        logger.debug("Using OPEN-ENDED prompt for multi-hop generation.")

    system_msg = {"role": "system", "content": system_prompt}

    dataset = custom_load_dataset(config=config, subset="chunked")
    logger.info(f"Loaded {len(dataset)} documents for multi-hop.")

    inference_calls, inference_index_map = build_multi_hop_inference_calls(dataset, system_msg, stage_cfg)
    if not inference_calls:
        logger.warning("No valid multi-hop chunks found for inference.")
        return

    responses = run_inference(config=config, step_name=MULTI_HOP_KEY, inference_calls=inference_calls)
    final_rows = parse_multi_hop_responses(responses, inference_index_map, stage_cfg)

    if final_rows:
        logger.info(f"Saving {len(final_rows)} multi-hop questions.")
        custom_save_dataset(Dataset.from_list(final_rows), config=config, subset="multi_hop_questions")
