"""Pipeline orchestrator for Yourbench."""

import time
import importlib
from functools import cache

from loguru import logger

# Lazy imports for heavy modules
_dataset_engine_loaded = False
_config_engine_loaded = False
_question_gen_loaded = False

def _lazy_load_dataset_engine():
    global _dataset_engine_loaded, upload_dataset_card
    if not _dataset_engine_loaded:
        logger.debug("Loading dataset engine...")
        from yourbench.utils.dataset_engine import upload_dataset_card
        _dataset_engine_loaded = True
    return upload_dataset_card

def _lazy_load_config_engine():
    global _config_engine_loaded, PipelineConfig, YourbenchConfig
    if not _config_engine_loaded:
        logger.debug("Loading configuration engine...")
        from yourbench.utils.configuration_engine import PipelineConfig, YourbenchConfig
        _config_engine_loaded = True
    return PipelineConfig, YourbenchConfig

def _lazy_load_question_gen():
    global _question_gen_loaded, run_multi_hop, run_single_shot, run_cross_document
    if not _question_gen_loaded:
        logger.debug("Loading question generation modules...")
        from yourbench.pipeline.question_generation import run_multi_hop, run_single_shot, run_cross_document
        _question_gen_loaded = True
    return run_multi_hop, run_single_shot, run_cross_document


@cache
def get_stage_order():
    PipelineConfig, _ = _lazy_load_config_engine()
    return PipelineConfig.STAGE_ORDER

@cache
def get_stage_overrides():
    run_multi_hop, run_single_shot, run_cross_document = _lazy_load_question_gen()
    return {
        "single_shot_question_generation": run_single_shot,
        "multi_hop_question_generation": run_multi_hop,
        "cross_document_question_generation": run_cross_document,
    }


@cache
def get_stage_function(stage: str):
    stage_overrides = get_stage_overrides()
    if func := stage_overrides.get(stage):
        return func

    # Support older configs
    # TODO add warning here
    if stage == "lighteval":
        logger.warning(f"Found depricated name for {stage}. Please, update your config to use 'prepare_lighteval'")
        stage = "prepare_lighteval"
    module = importlib.import_module(f"yourbench.pipeline.{stage}")
    return module.run


def run_stage(stage: str, config) -> float:
    logger.info(f"Running {stage}")
    start = time.perf_counter()

    try:
        get_stage_function(stage)(config)
        return time.perf_counter() - start
    except Exception:
        logger.exception(f"Error in {stage}")
        raise


def run_pipeline(config_file_path: str, debug: bool = False, **kwargs) -> None:
    _, YourbenchConfig = _lazy_load_config_engine()
    config = YourbenchConfig.from_yaml(config_file_path)
    config.debug = debug
    pipeline = config.pipeline_config

    if not pipeline:
        logger.warning("No pipeline stages configured")
        return

    stage_order = get_stage_order()
    for stage in stage_order:
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
        upload_dataset_card = _lazy_load_dataset_engine()
        upload_dataset_card(config)
    except Exception as e:
        logger.warning(f"Failed to upload dataset card: {e}")
