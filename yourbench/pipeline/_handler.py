# yourbench/pipeline/_handler.py

import os
import time
import matplotlib.pyplot as plt
from typing import Dict, Any
from yourbench.utils.loading_engine import load_config
from loguru import logger
import importlib

# 1) Create a global list to record pipeline-stage timings
GLOBAL_PIPELINE_TIMINGS = []  # Each item: {"stage_name": str, "start": float, "end": float}

def run_pipeline(config_file_path: str) -> None:
    """Execute the pipeline stages defined in the configuration file."""
    try:
        logger.debug(f"Loading configuration from {config_file_path}")
        config: Dict[str, Any] = load_config(config_file_path)
        pipeline_configuration = config.get("pipeline", {})

        if not pipeline_configuration:
            logger.warning("No pipeline stages configured")
            return

        # Make sure we have a 'plots' directory
        os.makedirs("plots", exist_ok=True)

        pipeline_start_time = time.time()

        # Execute pipeline stages
        for stage_name, stage_configuration in pipeline_configuration.items():
            try:
                if not stage_configuration.get("run", True):
                    logger.info(f"Skipping stage: {stage_name}")
                    continue

                logger.info(f"Starting execution of stage: {stage_name}")
                stage_module = importlib.import_module(f"yourbench.pipeline.{stage_name}")

                # 2) Time the stage
                t0 = time.time()
                stage_module.run(config)
                t1 = time.time()

                logger.success(f"Successfully completed stage: {stage_name}")
                GLOBAL_PIPELINE_TIMINGS.append({
                    "stage_name": stage_name,
                    "start": t0,
                    "end": t1,
                })
            except Exception as e:
                logger.error(f"Error executing pipeline stage {stage_name}: {str(e)}")
                raise

        pipeline_end_time = time.time()

        # 3) Produce final waterfall chart after all stages
        _plot_pipeline_stage_timing(GLOBAL_PIPELINE_TIMINGS, pipeline_start_time, pipeline_end_time)

    except Exception as e:
        logger.critical(f"Pipeline execution failed: {str(e)}")
        raise


def _plot_pipeline_stage_timing(
    stage_timings: list,
    pipeline_start: float,
    pipeline_end: float
) -> None:
    """
    Generate and save a waterfall (or simple bar) chart showing how long each
    pipeline stage took, plus total pipeline duration. Saves to 'plots/pipeline_stages_timing.png' (300 dpi).
    """
    if not stage_timings:
        logger.warning("No stage timings recorded. Skipping pipeline timing plot.")
        return

    # Sort by actual run order (each entry is appended after the stage completes).
    # If you want to preserve config order, you can skip sorting.
    # We assume stage_timings is in the order of completion, but let's keep it stable:
    # stage_timings is already appended in the same order we run them, so no sort needed.

    stage_names = []
    durations = []
    for entry in stage_timings:
        stage_name = entry["stage_name"]
        start = entry["start"]
        end = entry["end"]
        elapsed = end - start

        stage_names.append(stage_name)
        durations.append(elapsed)

    total_pipeline_time = pipeline_end - pipeline_start

    logger.info("Pipeline total duration = {:.2f} seconds.", total_pipeline_time)
    logger.info("Stage-by-stage breakdown:")
    for name, dur in zip(stage_names, durations):
        logger.info("  Stage '{}' took {:.2f} seconds", name, dur)

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(8, 5))
    # Simple bar chart approach:
    bars = ax.bar(stage_names, durations, color="royalblue")
    
    # Add duration text on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02 * max(durations),
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
    
    # Add a little padding at the top to make room for the labels
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    
    out_path = "plots/pipeline_stages_timing.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    logger.success(f"Pipeline timing chart saved to '{out_path}' (300 dpi).")
