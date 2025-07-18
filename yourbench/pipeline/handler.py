"""Pipeline orchestrator for Yourbench."""

import time
import importlib
from functools import cache

from loguru import logger

from yourbench.utils.dataset_engine import upload_dataset_card
from yourbench.utils.configuration_engine import YourbenchConfig
from yourbench.pipeline.question_generation import run_multi_hop, run_single_shot, run_cross_document


STAGE_ORDER = [
    "ingestion",
    "summarization",
    "chunking",
    "single_shot_question_generation",
    "multi_hop_question_generation",
    "cross_document_question_generation",
    "question_rewriting",
    "prepare_lighteval",
    "citation_score_filtering",
]

STAGE_OVERRIDES = {
    "single_shot_question_generation": run_single_shot,
    "multi_hop_question_generation": run_multi_hop,
    "cross_document_question_generation": run_cross_document,
}


@cache
def get_stage_function(stage: str):
    if func := STAGE_OVERRIDES.get(stage):
        return func
    module = importlib.import_module(f"yourbench.pipeline.{stage}")
    return module.run


def run_stage(stage: str, config: YourbenchConfig) -> float:
    logger.info(f"Running {stage}")
    start = time.perf_counter()

    try:
        get_stage_function(stage)(config)
        return time.perf_counter() - start
    except Exception:
        logger.exception(f"Error in {stage}")
        raise


def run_pipeline(config_file_path: str, debug: bool = False, **kwargs) -> None:
    config = YourbenchConfig.from_yaml(config_file_path)
    config.debug = debug
    pipeline = config.pipeline_config

    if not pipeline:
        logger.warning("No pipeline stages configured")
        return

    for stage in STAGE_ORDER:
        if not hasattr(pipeline, stage):
            logger.warning(f"Stage '{stage}' not configured in pipeline")
            continue

        stage_config = getattr(pipeline, stage)
        if not stage_config.run:
            logger.info(f"Skipping {stage} (disabled)")
            continue

        elapsed = run_stage(stage, config)
        logger.success(f"Completed {stage} in {elapsed:.3f}s")

    try:
        upload_dataset_card(config)
    except Exception as e:
        logger.warning(f"Failed to upload dataset card: {e}")
