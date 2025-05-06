# handler.py
# =============================================================================
# Author: @sumukshashidhar
#
# This module orchestrates the YourBench pipeline stages in a specified order.
# It reads pipeline configuration from a config dictionary, runs each stage
# if enabled, times each stage's execution, logs errors to stage-specific
# log files, and finally generates an overall timing chart of all stages.
#
# Usage:
#     from yourbench.pipeline.handler import run_pipeline
#     run_pipeline("/path/to/config.yaml", debug=True)
#
# The module assumes the presence of pipeline stages named after their .py
# files (e.g., ingestion, summarization), each exposing a `run(config: dict)`.
#
# Stages are executed in a fixed default order but will skip any that
# do not appear in the config or are explicitly disabled. Unrecognized
# stages in the config are also noted (but not executed).
#
# Key Responsibilities:
# 1. Load the user's pipeline configuration.
# 2. Execute each stage in `DEFAULT_STAGE_ORDER` if `run` is True in the config.
# 3. Log all events, including errors, to a stage-specific file and the console.
# 4. Collect and display timing data for each stage.
# 5. Detect any extra pipeline stages in the config that do not appear in
#    `DEFAULT_STAGE_ORDER` and log a warning about them.
# =============================================================================

from __future__ import annotations
import os
import time
import importlib
from typing import Any, Dict, List

from loguru import logger

from yourbench.utils.loading_engine import load_config


# === Pipeline Stage Order Definition ===
DEFAULT_STAGE_ORDER: List[str] = [
    "ingestion",
    "upload_ingest_to_hub",
    "summarization",
    "chunking",
    "single_shot_question_generation",
    "multi_hop_question_generation",
    # "deduplicate_single_shot_questions", #TODO: either remove or uncomment when implemented
    # "deduplicate_multi_hop_questions",
    "lighteval",
    "citation_score_filtering",
]

# This global list tracks the timing for all executed stages in the pipeline.
PIPELINE_STAGE_TIMINGS: List[Dict[str, float]] = []


def run_pipeline(
    config_file_path: str,
    debug: bool = False,
    plot_stage_timing: bool = False,
) -> None:
    """
    Run the YourBench pipeline based on a provided YAML/JSON configuration file.

    Args:
        config_file_path (str):
            Path to the pipeline configuration file that describes which stages to run (YAML or JSON).
        debug (bool):
            Enables more verbose logging (debug-level). Defaults to False.
        plot_stage_timing (bool):
            If True, generate a bar chart showing the time spent in each stage. Requires matplotlib.

    Raises:
        FileNotFoundError:
            If the configuration file is not found at the specified path.
        Exception:
            If any stage raises an unexpected error during execution, it is re-raised after logging.
    """
    global PIPELINE_STAGE_TIMINGS
    PIPELINE_STAGE_TIMINGS = []

    # Log level adjustments
    logger.debug(f"Loading pipeline configuration from {config_file_path}")
    config: Dict[str, Any] = load_config(config_file_path)

    # Attach debug flag to config for use in other modules
    config["debug"] = debug
    logger.info(f"Debug mode set to {config['debug']}")

    # Extract pipeline portion of the config
    pipeline_config: Dict[str, Any] = config.get("pipeline", {})
    if not pipeline_config:
        logger.warning("No pipeline stages configured. Exiting pipeline execution.")
        return

    # Ensure logs directory exists to store stage-specific logs
    os.makedirs("logs", exist_ok=True)

    # Record overall pipeline start
    pipeline_execution_start_time: float = time.time()

    # === Execute pipeline stages in the fixed default order ===
    for stage_name in DEFAULT_STAGE_ORDER:
        # Check if the stage is mentioned in the pipeline config at all
        if stage_name not in pipeline_config:
            logger.debug(f"Stage '{stage_name}' is not mentioned in the config. Skipping.")
            continue

        # Get the settings for the stage. It might be None or a dict.
        stage_settings = pipeline_config.get(stage_name)
        if not isinstance(stage_settings, dict):
            pipeline_config[stage_name] = {"run": True}
        elif "run" not in stage_settings:
            pipeline_config[stage_name]["run"] = True

        if not pipeline_config[stage_name]["run"]:
            logger.info(f"Skipping stage: '{stage_name}' (run set to False).")
            continue

        # Setup a stage-specific error log file
        error_log_path = os.path.join("logs", f"pipeline_{stage_name}.log")
        log_id = logger.add(error_log_path, level="ERROR", backtrace=True, diagnose=True, mode="a")

        logger.info(f"Starting execution of stage: '{stage_name}'")
        stage_start_time: float = time.time()

        # Ensure the specific stage config is at least an empty dict if it was None
        if stage_name in config.get("pipeline", {}) and config["pipeline"][stage_name] is None:
            config["pipeline"][stage_name] = {}

        try:
            # Dynamically import the stage module, e.g. yourbench.pipeline.ingestion
            stage_module = importlib.import_module(f"yourbench.pipeline.{stage_name}")
            stage_module.run(config)
        except Exception as pipeline_error:
            logger.error(f"Error executing pipeline stage '{stage_name}': {str(pipeline_error)}")
            # Remove stage-specific log handler before re-raising
            _remove_log_handler_safely(log_id)
            raise
        finally:
            _remove_log_handler_safely(log_id)

        stage_end_time: float = time.time()
        elapsed_time: float = stage_end_time - stage_start_time
        PIPELINE_STAGE_TIMINGS.append({
            "stage_name": stage_name,
            "start": stage_start_time,
            "end": stage_end_time,
            "elapsed": elapsed_time,
        })
        logger.success(f"Completed stage: '{stage_name}' in {elapsed_time:.3f}s")

    # Record overall pipeline end
    pipeline_execution_end_time: float = time.time()

    # Check for unrecognized stages in config
    _check_for_unrecognized_stages(pipeline_config)

    # Optionally plot pipeline stage timings
    if plot_stage_timing or debug:
        _plot_pipeline_stage_timing(
            pipeline_start=pipeline_execution_start_time,
            pipeline_end=pipeline_execution_end_time,
        )


def _check_for_unrecognized_stages(pipeline_config: Dict[str, Any]) -> None:
    """
    Warn about pipeline stages that exist in the config but
    are not in DEFAULT_STAGE_ORDER.

    Args:
        pipeline_config (Dict[str, Any]):
            The pipeline configuration dict (subset of the main config).
    """
    for stage in pipeline_config.keys():
        if stage not in DEFAULT_STAGE_ORDER:
            logger.warning(f"Unrecognized stage '{stage}' is present in config but not in DEFAULT_STAGE_ORDER.")


def _plot_pipeline_stage_timing(
    pipeline_start: float,
    pipeline_end: float,
) -> None:
    """
    Generate a bar chart illustrating the stage timings for the entire pipeline.

    Args:
        pipeline_start (float):
            Timestamp when the pipeline started.
        pipeline_end (float):
            Timestamp when the pipeline ended.
    """
    logger.info("Generating pipeline stage timing chart.")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Cannot generate timing chart: matplotlib is not installed.")
        return

    # Gather data
    stages = [timing["stage_name"] for timing in PIPELINE_STAGE_TIMINGS]
    durations = [timing["elapsed"] for timing in PIPELINE_STAGE_TIMINGS]

    # Minimalistic bar chart
    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax.barh(stages, durations, color="skyblue", edgecolor="black")

    ax.set_xlabel("Duration (s)")
    ax.set_title("Pipeline Stage Timings")

    # Annotate each bar with the stage's duration
    for i, duration in enumerate(durations):
        ax.text(duration + 0.01, i, f"{duration:.2f}s", va="center", fontsize=6)

    plt.tight_layout()
    plt.savefig("pipeline_stage_timing.png", dpi=300)
    plt.close(fig)
    logger.info("Saved pipeline stage timing chart to 'pipeline_stage_timing.png'.")


def _remove_log_handler_safely(log_id: int) -> None:
    """
    Remove a log handler (by log_id) from loguru, swallowing any ValueError
    if the handler is already removed or doesn't exist.

    Args:
        log_id (int):
            The handler ID returned by logger.add().
    """
    try:
        logger.remove(log_id)
    except ValueError:
        pass
