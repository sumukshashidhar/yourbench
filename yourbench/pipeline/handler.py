"""Pipeline orchestrator for Yourbench."""

import time
import importlib

from loguru import logger

from yourbench.conf.loader import get_enabled_stages


def _get_stage_function(stage: str):
    """Get the function for a pipeline stage."""
    # Handle legacy name
    if stage == "lighteval":
        logger.warning("'lighteval' is deprecated, use 'prepare_lighteval'")
        stage = "prepare_lighteval"

    module = importlib.import_module(f"yourbench.pipeline.{stage}")
    return module.run


def run_stage(stage: str, config) -> float:
    """Run a single pipeline stage, return elapsed time."""
    logger.info(f"Running {stage}")
    start = time.perf_counter()
    try:
        _get_stage_function(stage)(config)
        return time.perf_counter() - start
    except Exception:
        logger.exception(f"Error in {stage}")
        raise


def run_pipeline(config_path: str, debug: bool = False) -> None:
    """Run the full pipeline from a config file path."""
    from yourbench.conf.loader import load_config

    config = load_config(config_path)
    if debug:
        config.debug = True

    run_pipeline_with_config(config, debug=debug)


def run_pipeline_with_config(config, debug: bool = False) -> None:
    """Run the pipeline with a pre-loaded config object."""
    if debug:
        config.debug = True

    enabled = get_enabled_stages(config)
    if not enabled:
        logger.warning("No pipeline stages enabled")
        return

    logger.info(f"Running stages: {', '.join(enabled)}")

    for stage in enabled:
        elapsed = run_stage(stage, config)
        logger.success(f"Completed {stage} in {elapsed:.2f}s")

    # Upload dataset card
    try:
        from yourbench.utils.dataset_card import upload_dataset_card

        upload_dataset_card(config)
    except Exception as e:
        logger.warning(f"Failed to upload dataset card: {e}")
