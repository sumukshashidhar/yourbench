"""Pipeline orchestrator for Yourbench."""

import time
import importlib
from functools import cache

from loguru import logger

from yourbench.utils.dataset_engine import upload_dataset_card

# Import stage order from configuration for consistency
from yourbench.utils.configuration_engine import PipelineConfig, YourbenchConfig
from yourbench.pipeline.question_generation import run_multi_hop, run_single_shot, run_cross_document


STAGE_ORDER = PipelineConfig.STAGE_ORDER

STAGE_OVERRIDES = {
    "single_shot_question_generation": run_single_shot,
    "multi_hop_question_generation": run_multi_hop,
    "cross_document_question_generation": run_cross_document,
}


@cache
def get_stage_function(stage: str):
    if func := STAGE_OVERRIDES.get(stage):
        return func
    
    # Support older configs
    # TODO add warning here
    if stage == "lighteval":
        logger.warning(f"Found depricated name for {stage}. Please, update your config to use 'prepare_lighteval'")
        stage = "prepare_lighteval"
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
        try:
            stage_config = pipeline.get_stage_config(stage)
        except ValueError:
            logger.warning(f"Stage '{stage}' not configured in pipeline")
            continue

        if not stage_config.run:
            logger.info(f"Skipping {stage} (disabled)")
            continue

        elapsed = run_stage(stage, config)
        logger.success(f"Completed {stage} in {elapsed:.3f}s")

    try:
        upload_dataset_card(config)
    except Exception as e:
        logger.warning(f"Failed to upload dataset card: {e}")
