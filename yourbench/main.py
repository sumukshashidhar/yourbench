import argparse
import sys
from loguru import logger

from config.pipeline_steps import PIPELINE_STEPS
from utils.load_task_config import get_available_tasks, load_task_config
from interface.frontend import launch_frontend

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
    Execute a single pipeline step based on the provided configuration.

    :param config: Task configuration dictionary
    :param step_name: Name of the pipeline step to execute
    """
    try:
        logger.info(f"Starting {PIPELINE_STEPS[step_name]['description']}...")
        PIPELINE_STEPS[step_name]["func"](config)
        logger.info(f"Completed {PIPELINE_STEPS[step_name]['description']}.")
    except Exception as e:
        logger.exception(f"Error in {PIPELINE_STEPS[step_name]['description']}: {e}")

def process_pipeline(config: dict) -> None:
    """
    Execute the pipeline steps specified in the task configuration.

    :param config: Task configuration dictionary
    """
    executed_steps = []
    skipped_steps = []

    for step_name, step_config in config["pipeline"].items():
        if step_name not in PIPELINE_STEPS:
            logger.warning(f"Unknown step: {step_name}. Skipping...")
            skipped_steps.append(step_name)
            continue

        if step_config.get("execute", False):
            logger.debug(f"Executing step: {step_name}")
            executed_steps.append(step_name)
            process_pipeline_step(config, step_name)
        else:
            skipped_steps.append(step_name)
            logger.debug(
                f"Skipping step: {PIPELINE_STEPS[step_name]['description']} (not specified in the task config)"
            )

    # Log pipeline execution summary
    logger.info("Pipeline processing completed.")
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
    parser.add_argument(
        "--task-name", dest="task_name_opt", help="Name of the task to process"
    )
    # Flag to launch the frontend interface
    parser.add_argument(
        "--frontend", action="store_true", help="Launch the Gradio frontend interface"
    )
    args = parser.parse_args()

    # Check if frontend should be launched
    if args.frontend:
        logger.info("Launching frontend interface...")
        launch_frontend()
        return

    task_name = args.task_name or args.task_name_opt
    if not task_name:
        # Raise an error if no task name is provided
        parser.error(
            "Task name must be provided either as a positional argument or with --task-name"
        )

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

    # Start pipeline processing
    logger.info("Beginning pipeline processing...", task=task_name)
    process_pipeline(config)
    logger.success("Pipeline processing completed for task: {}", task_name)


if __name__ == "__main__":
    main()
