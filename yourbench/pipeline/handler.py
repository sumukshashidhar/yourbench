"""Pipeline orchestrator for Yourbench."""

import time
import importlib
from functools import cache

from loguru import logger
from yourbench.utils.configuration_engine import PipelineConfig, YourbenchConfig  # noqa: E402


# Lazy imports for heavy modules
_dataset_engine_loaded = False
_upload_dataset_card = None


def _lazy_load_dataset_engine():
    global _dataset_engine_loaded, _upload_dataset_card
    if not _dataset_engine_loaded:
        from yourbench.utils.dataset_engine import upload_dataset_card

        _upload_dataset_card = upload_dataset_card
        _dataset_engine_loaded = True
    return _upload_dataset_card


# Lazy imports for question generation
_qg_loaded = False
_run_multi_hop = None
_run_single_shot = None
_run_cross_document = None


def _lazy_load_question_generation():
    global _qg_loaded, _run_multi_hop, _run_single_shot, _run_cross_document
    if not _qg_loaded:
        from yourbench.pipeline.question_generation import run_multi_hop, run_single_shot, run_cross_document

        _run_multi_hop = run_multi_hop
        _run_single_shot = run_single_shot
        _run_cross_document = run_cross_document
        _qg_loaded = True
    return _run_multi_hop, _run_single_shot, _run_cross_document


STAGE_ORDER = PipelineConfig.STAGE_ORDER


def _get_stage_overrides():
    """Get stage overrides with lazy loading."""
    run_multi_hop, run_single_shot, run_cross_document = _lazy_load_question_generation()
    return {
        "single_shot_question_generation": run_single_shot,
        "multi_hop_question_generation": run_multi_hop,
        "cross_document_question_generation": run_cross_document,
    }


# For backward compatibility with tests
STAGE_OVERRIDES = _get_stage_overrides()


@cache
def get_stage_function(stage: str):
    overrides = _get_stage_overrides()
    if func := overrides.get(stage):
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
        upload_func = _lazy_load_dataset_engine()
        upload_func(config)
    except Exception as e:
        logger.warning(f"Failed to upload dataset card: {e}")
