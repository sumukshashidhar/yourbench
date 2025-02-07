import argparse
import sys
import time

from config.pipeline_steps import PIPELINE_STEPS
from interface.frontend import launch_frontend
from loguru import logger
from utils.load_task_config import get_available_tasks, load_task_config


# Configure Loguru for logging and error handling
logger.remove()  # Remove any default handlers
logger.add(
    sys.stderr,
    level="INFO",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
)


def process_pipeline_step(config: dict, step_name: str) -> None:
    """
    Execute a single pipeline step and measure execution time.

    :param config: Task configuration dictionary
    :param step_name: Name of the pipeline step to execute
    """
    try:
        logger.info(f"Starting {PIPELINE_STEPS[step_name]['description']}...")
        start_time = time.time()

        PIPELINE_STEPS[step_name]["func"](config)

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Completed {PIPELINE_STEPS[step_name]['description']} in {duration:.2f} seconds.")
    except Exception as e:
        logger.exception(f"Error in {PIPELINE_STEPS[step_name]['description']}: {e}")


def process_pipeline(config: dict) -> None:
    """
    Execute the pipeline steps specified in the task configuration and log execution time.

    :param config: Task configuration dictionary
    """
    executed_steps = []
    skipped_steps = []
    step_times = {}

    pipeline_start = time.time()

    for step_name, step_config in config["pipeline"].items():
        if step_name not in PIPELINE_STEPS:
            logger.warning(f"Unknown step: {step_name}. Skipping...")
            skipped_steps.append(step_name)
            continue

        if step_config.get("execute", False):
            logger.debug(f"Executing step: {step_name}")
            executed_steps.append(step_name)

            step_start = time.time()
            process_pipeline_step(config, step_name)
            step_end = time.time()

            step_times[step_name] = step_end - step_start
        else:
            skipped_steps.append(step_name)
            logger.debug(
                f"Skipping step: {PIPELINE_STEPS[step_name]['description']} (not specified in the task config)"
            )

    pipeline_end = time.time()
    total_duration = pipeline_end - pipeline_start

    # Log step-wise execution time
    for step, duration in step_times.items():
        logger.info(f"Step '{step}' execution time: {duration:.2f} seconds")

    # Log pipeline execution summary
    logger.info("Pipeline processing completed.")
    logger.info("Total execution time: {:.2f} seconds".format(total_duration))
    logger.info("Executed steps: {}", ", ".join(executed_steps) if executed_steps else "None")
    logger.info("Skipped steps: {}", ", ".join(skipped_steps) if skipped_steps else "None")


def main() -> None:
    """
    Main entry point for the script.

    Parse command-line arguments and execute the pipeline or launch the frontend.
    """
    parser = argparse.ArgumentParser(description="Process a specific YourBench task.")
    # Positional argument for task name
    parser.add_argument("task_name", nargs="?", help="Name of the task to process")
    # Optional argument for task name (alternative to positional)
    parser.add_argument("--task-name", dest="task_name_opt", help="Name of the task to process")
    # Flag to launch the frontend interface
    parser.add_argument("--frontend", action="store_true", help="Launch the Gradio frontend interface")
    # Flag to concatenate datasets if they exist
    parser.add_argument("--concat", action="store_true", help="Concatenate existing dataset with new data")

    args = parser.parse_args()

    # Check if frontend should be launched
    if args.frontend:
        logger.info("Launching frontend interface...")
        launch_frontend()
        return

    task_name = args.task_name or args.task_name_opt
    if not task_name:
        # Raise an error if no task name is provided
        parser.error("Task name must be provided either as a positional argument or with --task-name")

    available_tasks = get_available_tasks()
    if task_name not in available_tasks:
        # Log and exit on invalid task
        logger.error(
            "Invalid task: {}. Available tasks: {}",
            task_name,
            ", ".join(available_tasks),
        )
        sys.exit(1)

    logger.info("Loading configuration for task: {}", task_name)
    config = load_task_config(task_name)
    logger.debug(f"Loaded task configuration: {config}")

    # Retrieve `concat_if_exists` from config
    concat_if_exists = config.get("configurations", {}).get("huggingface", {}).get("concat_if_exists", False)
    config["concat_datasets"] = concat_if_exists

    # Start pipeline processing
    logger.info("Beginning pipeline processing...", task=task_name)
    process_pipeline(config)
    logger.success("Pipeline processing completed for task: {}", task_name)


if __name__ == "__main__":
    main()
