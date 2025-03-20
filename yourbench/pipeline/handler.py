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

import os
import time
import importlib
import matplotlib.pyplot as plt
from typing import Dict, Any, List
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
    "deduplicate_single_shot_questions",
    "deduplicate_multi_hop_questions",
    "lighteval"
]

# This global list tracks the timing for all executed stages in the pipeline.
PIPELINE_STAGE_TIMINGS: List[Dict[str, float]] = []


def run_pipeline(config_file_path: str, debug: bool = False) -> None:
    """
    Run the YourBench pipeline based on a provided YAML configuration file.

    This function:
      1. Loads the pipeline configuration from `config_file_path`.
      2. Iterates over the `DEFAULT_STAGE_ORDER` and executes each stage
         if it is present and enabled (`run = True`) in the config.
      3. Logs errors for each stage to a dedicated file in `logs/` named
         `pipeline_<stage_name>.log`.
      4. Records the start and end time of each stage to produce a timing chart.
      5. Logs warnings for any stages in the config that are not part
         of `DEFAULT_STAGE_ORDER`.

    Args:
        config_file_path (str):
            Path to the YAML configuration file that describes how
            the pipeline should run (e.g., which stages to enable).
        debug (bool):
            Indicates whether to run in debug mode (more verbose logging).

    Raises:
        FileNotFoundError:
            If the configuration file is not found at the specified path.
        Exception:
            If any stage raises an unexpected error during execution, the
            error is re-raised after being logged.
    """
    global PIPELINE_STAGE_TIMINGS
    PIPELINE_STAGE_TIMINGS = []

    # Load the pipeline config from the specified file
    logger.debug(f"Loading pipeline configuration from {config_file_path}")
    config: Dict[str, Any] = load_config(config_file_path)

    # Attach debug flag to config for use in other modules
    config["debug"] = debug
    logger.info(f"Debug mode set to {config['debug']}")

    # Extract the pipeline portion of the config
    pipeline_config: Dict[str, Any] = config.get("pipeline", {})
    if not pipeline_config:
        logger.warning("No pipeline stages configured. Exiting pipeline execution.")
        return

    # Ensure logs directory exists to store stage-specific logs
    os.makedirs("logs", exist_ok=True)

    # Record the overall pipeline start time
    pipeline_execution_start_time: float = time.time()

    # === Execute pipeline stages in the fixed default order ===
    for stage_name in DEFAULT_STAGE_ORDER:
        stage_config = pipeline_config.get(stage_name)
        if stage_config is None:
            logger.debug(f"Stage '{stage_name}' is not present in the config. Skipping.")
            continue

        if not stage_config.get("run", True):
            logger.info(f"Skipping stage: '{stage_name}' (run: false).")
            continue

        # Set up a stage-specific error log file
        error_log_path = os.path.join("logs", f"pipeline_{stage_name}.log")
        log_id = logger.add(
            error_log_path, level="ERROR", backtrace=True, diagnose=True, mode="a"
        )

        logger.info(f"Starting execution of stage: '{stage_name}'")
        stage_start_time: float = time.time()

        try:
            # Dynamically import the module for this stage and run
            stage_module = importlib.import_module(f"yourbench.pipeline.{stage_name}")
            stage_module.run(config)
        except Exception as pipeline_error:
            logger.error(f"Error executing pipeline stage '{stage_name}': {str(pipeline_error)}")
            # Safely remove stage-specific log file handler before re-raising
            try:
                logger.remove(log_id)
            except ValueError:
                # Handler was already removed or doesn't exist, which is fine
                pass
            raise
        finally:
            # Safely remove the stage-specific error log handler
            try:
                logger.remove(log_id)
            except ValueError:
                # Handler was already removed or doesn't exist, which is fine
                pass

        stage_end_time: float = time.time()
        elapsed_time: float = stage_end_time - stage_start_time
        PIPELINE_STAGE_TIMINGS.append({
            "stage_name": stage_name,
            "start": stage_start_time,
            "end": stage_end_time
        })

        logger.success(
            f"Successfully completed stage: '{stage_name}' in {elapsed_time:.3f} seconds"
        )

    # Record the overall pipeline end time
    pipeline_execution_end_time: float = time.time()

    # Generate a bar chart illustrating the stage timings
    _plot_pipeline_stage_timing(
        stage_timings=PIPELINE_STAGE_TIMINGS,
        pipeline_start=pipeline_execution_start_time,
        pipeline_end=pipeline_execution_end_time,
    )

    # Handle any unrecognized pipeline stages in the config
    _handle_unordered_stages(
        pipeline_config=pipeline_config,
        ordered_stages=DEFAULT_STAGE_ORDER
    )
    logger.success(f"Please visit https://huggingface.co/datasets/{config['hf_configuration']['global_dataset_name']} to view your results!")


def _plot_pipeline_stage_timing(
    stage_timings: List[Dict[str, float]],
    pipeline_start: float,
    pipeline_end: float
) -> None:
    """
    Create and save a bar chart of pipeline stage durations as
    `plots/pipeline_stages_timing.png`.

    Args:
        stage_timings (List[Dict[str, float]]):
            A list of dictionaries with 'stage_name', 'start', and 'end' keys.
        pipeline_start (float):
            The wall-clock time that the pipeline began execution.
        pipeline_end (float):
            The wall-clock time that the pipeline ended execution.

    Returns:
        None
    """
    if not stage_timings:
        logger.warning("No stage timings recorded. Skipping pipeline timing plot.")
        return

    import numpy as np

    stage_names: List[str] = []
    durations: List[float] = []

    for entry in stage_timings:
        s_name: str = entry["stage_name"]
        s_start: float = entry["start"]
        s_end: float = entry["end"]
        s_elapsed: float = s_end - s_start
        stage_names.append(s_name)
        durations.append(s_elapsed)

    total_pipeline_time: float = pipeline_end - pipeline_start
    logger.info(f"Pipeline total duration = {total_pipeline_time:.2f} seconds.")
    logger.info("Stage-by-stage breakdown:")

    for name, duration in zip(stage_names, durations):
        logger.info(f"  Stage '{name}' took {duration:.2f} seconds.")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(stage_names, durations, color="royalblue")

    max_duration: float = max(durations)
    for bar in bars:
        height: float = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02 * max_duration,
            f"{height:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.set_xlabel("Pipeline Stage")
    ax.set_ylabel("Duration (seconds)")
    ax.set_title("YourBench Pipeline Stage Timings")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

    os.makedirs("plots", exist_ok=True)
    plot_file_path: str = os.path.join("plots", "pipeline_stages_timing.png")
    plt.savefig(plot_file_path, dpi=300)
    plt.close(fig)

    logger.success(f"Pipeline timing chart saved to '{plot_file_path}' (300 dpi).")


def _handle_unordered_stages(
    pipeline_config: Dict[str, Any],
    ordered_stages: List[str]
) -> None:
    """
    Identify stages in `pipeline_config` that do not appear
    in `ordered_stages` and log a warning for each.

    Args:
        pipeline_config (Dict[str, Any]):
            Dictionary containing all pipeline configuration data, typically
            `config["pipeline"]` from the main config.
        ordered_stages (List[str]):
            The known set of pipeline stages recognized in the standard
            `DEFAULT_STAGE_ORDER`.

    Returns:
        None
    """
    for stage_name in pipeline_config.keys():
        if stage_name not in ordered_stages:
            logger.warning(
                f"Stage '{stage_name}' is not in the default order list and has been skipped."
            )