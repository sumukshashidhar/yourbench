"""
This module orchestrates the Yourbench pipeline stages in a specified order.
It reads pipeline configuration from a config dictionary, runs each stage
if enabled, times each stage's execution, logs errors to stage-specific
log files, and finally generates an overall timing chart of all stages.

The module assumes the presence of pipeline stages named after their .py
files (e.g., ingestion, summarization), each exposing a `run(config: dict)`.

Some stages may use direct function overrides (e.g., for question generation),
bypassing dynamic import. These are defined in `STAGE_FUNCTION_OVERRIDES`.

Stages are executed in a fixed default order but will skip any that
do not appear in the config or are explicitly disabled. Unrecognized
stages in the config are also noted (but not executed).

Key Responsibilities:
1. Load the user's pipeline configuration.
2. Execute each stage in `DEFAULT_STAGE_ORDER` if `run` is True in the config.
3. Use function overrides for specific stages if defined.
4. Log all events, including errors, to a stage-specific file and the console.
5. Collect and display timing data for each stage.
6. Detect any extra pipeline stages in the config that do not appear in
   `DEFAULT_STAGE_ORDER` and log a warning about them.
"""

from __future__ import annotations
import os
import time
import importlib
from typing import Any, Dict, List

from loguru import logger

from yourbench.utils.loading_engine import load_config
from yourbench.pipeline.question_generation import (
    run_multi_hop,
    run_single_shot,
)


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

PIPELINE_STAGE_TIMINGS: List[Dict[str, float]] = []


STAGE_FUNCTION_OVERRIDES = {
    "single_shot_question_generation": run_single_shot,
    "multi_hop_question_generation": run_multi_hop,
}


def run_pipeline(
    config_file_path: str,
    debug: bool = False,
    plot_stage_timing: bool = False,
) -> None:
    global PIPELINE_STAGE_TIMINGS
    PIPELINE_STAGE_TIMINGS = []

    logger.debug(f"Loading pipeline configuration from {config_file_path}")
    config: Dict[str, Any] = load_config(config_file_path)
    config["debug"] = debug
    logger.info(f"Debug mode set to {config['debug']}")

    pipeline_config: Dict[str, Any] = config.get("pipeline", {})
    if not pipeline_config:
        logger.warning("No pipeline stages configured. Exiting pipeline execution.")
        return

    os.makedirs("logs", exist_ok=True)
    pipeline_execution_start_time: float = time.time()

    for stage_name in DEFAULT_STAGE_ORDER:
        if stage_name not in pipeline_config:
            logger.debug(f"Stage '{stage_name}' is not mentioned in the config. Skipping.")
            continue

        stage_settings = pipeline_config.get(stage_name)
        if not isinstance(stage_settings, dict):
            pipeline_config[stage_name] = {"run": True}
        elif "run" not in stage_settings:
            pipeline_config[stage_name]["run"] = True

        if not pipeline_config[stage_name]["run"]:
            logger.info(f"Skipping stage: '{stage_name}' (run set to False).")
            continue

        error_log_path = os.path.join("logs", f"pipeline_{stage_name}.log")
        log_id = logger.add(error_log_path, level="ERROR", backtrace=True, diagnose=True, mode="a")

        logger.info(f"Starting execution of stage: '{stage_name}'")
        stage_start_time: float = time.time()

        try:
            stage_func = STAGE_FUNCTION_OVERRIDES.get(stage_name)
            if stage_func:
                stage_func(config)
            else:
                stage_module = importlib.import_module(f"yourbench.pipeline.{stage_name}")
                stage_module.run(config)
        except Exception:
            logger.exception(f"Error executing pipeline stage '{stage_name}'")
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

    pipeline_execution_end_time: float = time.time()
    _check_for_unrecognized_stages(pipeline_config)

    # Upload the dataset card
    try:
        from yourbench.utils.dataset_engine import upload_dataset_card

        logger.info("Uploading final dataset card with all pipeline stages information")
        upload_dataset_card(config)
    except Exception as e:
        logger.warning(f"Failed to upload final dataset card: {e}")

    if plot_stage_timing or debug:
        _plot_pipeline_stage_timing()


def _check_for_unrecognized_stages(pipeline_config: Dict[str, Any]) -> None:
    for stage in pipeline_config.keys():
        if stage not in DEFAULT_STAGE_ORDER:
            logger.warning(f"Unrecognized stage '{stage}' is present in config but not in DEFAULT_STAGE_ORDER.")


def _plot_pipeline_stage_timing() -> None:
    logger.info("Generating pipeline stage timing chart.")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Cannot generate timing chart: matplotlib is not installed.")
        return

    stages = [timing["stage_name"] for timing in PIPELINE_STAGE_TIMINGS]
    durations = [timing["elapsed"] for timing in PIPELINE_STAGE_TIMINGS]

    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax.barh(stages, durations, color="skyblue", edgecolor="black")

    ax.set_xlabel("Duration (s)")
    ax.set_title("Pipeline Stage Timings")

    for i, duration in enumerate(durations):
        ax.text(duration + 0.01, i, f"{duration:.2f}s", va="center", fontsize=6)

    plt.tight_layout()
    plt.savefig("pipeline_stage_timing.png", dpi=300)
    plt.close(fig)
    logger.info("Saved pipeline stage timing chart to 'pipeline_stage_timing.png'.")


def _remove_log_handler_safely(log_id: int) -> None:
    try:
        logger.remove(log_id)
    except ValueError:
        pass
