# yourbench/pipeline/_handler.py

import os
import time
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from yourbench.utils.loading_engine import load_config
from loguru import logger
import importlib
import time

# === Pipeline Stage Order Definition ===
# This list enforces the exact order in which pipeline stages are executed.
# Stages not present here will be skipped (with a warning logged).
DEFAULT_STAGE_ORDER: List[str] = [
    "ingestion",
    "upload_ingest_to_hub",
    "summarization",
    "chunking",
    "single_shot_question_generation",
    "multi_hop_question_generation"
]

# === Global Variable for Pipeline Stage Timings ===
GLOBAL_PIPELINE_TIMINGS: List[Dict[str, Any]] = []

def run_pipeline(config_file_path: str, debug: bool = False) -> None:
    """
    Execute the pipeline stages defined in the configuration file in the
    strictly defined order of `DEFAULT_STAGE_ORDER`.

    Args:
        config_file_path: Path to the configuration file containing pipeline settings.
        debug: Boolean flag indicating whether to enable debug mode.

    Raises:
        FileNotFoundError: If the config file cannot be found.
        Exception: For any other unexpected errors during pipeline execution.
    """
    # === Initialization ===
    global GLOBAL_PIPELINE_TIMINGS
    GLOBAL_PIPELINE_TIMINGS = []
    
    try:
        logger.debug(f"Loading configuration from {config_file_path}")
        config: Dict[str, Any] = load_config(config_file_path)

        # Inject the debug flag into the config
        config["debug"] = debug
        logger.info(f"Debug mode set to {config['debug']}")

        # Retrieve the pipeline configuration dictionary
        pipeline_configuration: Dict[str, Any] = config.get("pipeline", {})
        if not pipeline_configuration:
            logger.warning("No pipeline stages configured. Exiting pipeline execution.")
            return

        # Ensure a 'plots' directory exists for saving charts
        os.makedirs("plots", exist_ok=True)

        pipeline_start_time: float = time.time()

        # === Execute pipeline stages in the fixed order ===
        for stage_name in DEFAULT_STAGE_ORDER:
            stage_configuration = pipeline_configuration.get(stage_name)
            if stage_configuration is None:
                # This means the stage is not present in the config. We skip it.
                logger.debug(f"Stage '{stage_name}' is not present in the config. Skipping.")
                continue

            # Check if this stage is disabled
            if not stage_configuration.get("run", True):
                logger.info(f"Skipping stage: '{stage_name}' (run: false).")
                continue

            # === Stage Execution ===
            logger.info(f"Starting execution of stage: '{stage_name}'")
            start_time: float = time.time()
            
            try:
                # Dynamically import and run the module
                stage_module = importlib.import_module(f"yourbench.pipeline.{stage_name}")
                stage_module.run(config)
            except Exception as e:
                logger.error(f"Error executing pipeline stage '{stage_name}': {str(e)}")
                raise

            end_time: float = time.time()
            elapsed_time: float = end_time - start_time
            
            # Record the timing information
            GLOBAL_PIPELINE_TIMINGS.append({
                "stage_name": stage_name,
                "start": start_time,
                "end": end_time
            })
            
            logger.success(f"Successfully completed stage: '{stage_name}' in {elapsed_time:.3f} seconds")

        pipeline_end_time: float = time.time()

        # === Generate Waterfall Chart ===
        _plot_pipeline_stage_timing(
            stage_timings=GLOBAL_PIPELINE_TIMINGS,
            pipeline_start=pipeline_start_time,
            pipeline_end=pipeline_end_time
        )

        # === Handle any extra stages not in DEFAULT_STAGE_ORDER (if present) ===
        _handle_unordered_stages(
            pipeline_config=pipeline_configuration,
            ordered_stages=DEFAULT_STAGE_ORDER
        )

    except Exception as e:
        logger.critical(f"Pipeline execution failed: {str(e)}")
        raise


def _plot_pipeline_stage_timing(
    stage_timings: List[Dict[str, float]],
    pipeline_start: float,
    pipeline_end: float
) -> None:
    """
    Generate and save a bar chart (waterfall style) showing the duration of each
    pipeline stage. The chart is saved to 'plots/pipeline_stages_timing.png'.

    Args:
        stage_timings: A list of dictionaries containing stage timing data.
        pipeline_start: The start time of the overall pipeline.
        pipeline_end: The end time of the overall pipeline.
    """
    if not stage_timings:
        logger.warning("No stage timings recorded. Skipping pipeline timing plot.")
        return

    # We rely on the append order in GLOBAL_PIPELINE_TIMINGS, which matches
    # the actual run order from the strict pipeline ordering.
    stage_names: List[str] = []
    durations: List[float] = []

    for entry in stage_timings:
        stage_name: str = entry["stage_name"]
        start: float = entry["start"]
        end: float = entry["end"]
        elapsed: float = end - start

        stage_names.append(stage_name)
        durations.append(elapsed)

    total_pipeline_time: float = pipeline_end - pipeline_start

    logger.info("Pipeline total duration = {:.2f} seconds.", total_pipeline_time)
    logger.info("Stage-by-stage breakdown:")
    for name, dur in zip(stage_names, durations):
        logger.info("  Stage '{}' took {:.2f} seconds", name, dur)

    # === Plotting the bar chart ===
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(stage_names, durations, color="royalblue")
    
    # Add duration text on top of each bar
    max_duration: float = max(durations)
    for bar in bars:
        height: float = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02 * max_duration,
            f"{height:.2f}s",
            ha='center',
            va='bottom',
            fontsize=9
        )
        
    ax.set_xlabel("Pipeline Stage")
    ax.set_ylabel("Duration (seconds)")
    ax.set_title("YourBench Pipeline Stage Timings")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Add extra padding for the labels
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    
    out_path: str = "plots/pipeline_stages_timing.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    logger.success(f"Pipeline timing chart saved to '{out_path}' (300 dpi).")


def _handle_unordered_stages(pipeline_config: Dict[str, Any], ordered_stages: List[str]) -> None:
    """
    Check for stages in `pipeline_config` that are not in `ordered_stages`.
    Log a warning if any are found (since they're skipped).

    Args:
        pipeline_config: Dictionary containing pipeline configuration from YAML.
        ordered_stages: The list of known stages in the correct order.
    """
    for stage_name in pipeline_config.keys():
        if stage_name not in ordered_stages:
            logger.warning(
                f"Stage '{stage_name}' is not in the default order list and has been skipped."
            )
