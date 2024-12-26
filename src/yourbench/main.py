import argparse
import sys

from loguru import logger


# Configure Loguru
logger.remove()  # Remove any default handlers
logger.add(
    sys.stderr,
    level="DEBUG",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
)


def process_generate_dataset(config: dict) -> None:
    """Generate dataset based on the provided config."""
    from yourbench.preprocessing.dataset_generation import generate_dataset

    logger.info("Starting dataset generation...", step="generate_dataset")
    generate_dataset(config=config)


def process_generate_summaries(config: dict) -> None:
    """Generate summaries for documents based on the provided config."""
    from yourbench.preprocessing.generate_summaries import (
        generate_summaries_for_documents,
    )

    logger.info("Starting summary generation...", step="generate_summaries")
    generate_summaries_for_documents(config=config)


def process_create_chunks(config: dict) -> None:
    """Create chunks for documents based on the provided config."""
    from yourbench.preprocessing.create_chunks import create_chunks_for_documents

    logger.info("Starting chunk creation...", step="create_chunks")
    create_chunks_for_documents(config=config)


def process_make_multihop_chunks(config: dict) -> None:
    """Create multi-hop chunks for documents based on the provided config."""
    from yourbench.preprocessing.create_multihop_chunks import create_multihop_chunks

    logger.info("Starting multi-hop chunk creation...", step="make_multihop_chunks")
    create_multihop_chunks(config=config)


def process_create_single_shot_questions(config: dict) -> None:
    """Generate single-shot questions based on the provided config."""
    from yourbench.question_generation.generate_questions import (
        generate_single_shot_questions,
    )

    logger.info(
        "Starting single-shot question generation...",
        step="create_single_shot_questions",
    )
    generate_single_shot_questions(config=config)


def process_create_multihop_questions(config: dict) -> None:
    """Generate multi-hop questions based on the provided config."""
    from yourbench.question_generation.generate_questions import (
        generate_multihop_questions,
    )

    logger.info(
        "Starting multi-hop question generation...", step="create_multihop_questions"
    )
    generate_multihop_questions(config=config)


def process_reweight_and_deduplicate_questions(config: dict) -> None:
    """Reweight and deduplicate questions based on the provided config."""
    from yourbench.postprocessing.reweight_and_deduplication import (
        reweight_and_deduplicate_questions,
    )

    logger.info(
        "Starting question reweighting and deduplication...",
        step="reweight_and_deduplicate_questions",
    )
    reweight_and_deduplicate_questions(config=config)


def process_answer_questions_with_llm(config: dict) -> None:
    """Use an LLM to answer questions based on the provided config."""
    from yourbench.question_answering.answer_questions import answer_questions_with_llm

    logger.info(
        "Starting question answering with LLM...", step="answer_questions_with_llm"
    )
    answer_questions_with_llm(config=config)


def process_reformat_for_judging(config: dict) -> None:
    """Reformat answers for the judge."""
    from yourbench.postprocessing.reformat_dataset_for_judge import reformat_for_judging

    logger.info("Starting dataset reformat for judging...", step="reformat_for_judging")
    reformat_for_judging(config=config)


def process_judge_answers(config: dict) -> None:
    """Judge answers using the provided config."""
    from yourbench.judge.judge_answers import judge_answers

    logger.info("Starting answer judging...", step="judge")
    judge_answers(config=config)


def process_visualize_results(config: dict) -> None:
    """Visualize judge results."""
    from yourbench.visualizations.visualize_judge_results import visualize_judge_results

    logger.info("Starting results visualization...", step="visualize_results")
    visualize_judge_results(config=config)


def process_pipeline(config: dict) -> None:
    """
    Process the YourBench pipeline for a given task by executing each step,
    as specified in the config.
    """
    # Define the pipeline steps in an ordered list
    pipeline_steps = [
        ("generate_dataset", process_generate_dataset),
        ("generate_summaries", process_generate_summaries),
        ("create_chunks", process_create_chunks),
        ("make_chunk_pairings", process_make_multihop_chunks),
        ("create_single_hop_questions", process_create_single_shot_questions),
        ("create_multi_hop_questions", process_create_multihop_questions),
        (
            "reweight_and_deduplicate_questions",
            process_reweight_and_deduplicate_questions,
        ),
        ("answer_questions_with_llm", process_answer_questions_with_llm),
        ("reformat_for_judging", process_reformat_for_judging),
        ("judge", process_judge_answers),
        ("visualize_results", process_visualize_results),
    ]

    for step_name, step_func in pipeline_steps:
        should_execute = config["pipeline"].get(step_name, {}).get("execute", False)
        if should_execute:
            logger.debug("Executing step: {}", step_name)
            step_func(config)
        else:
            logger.debug(
                "Skipping step: {} (not specified in the task config)", step_name
            )


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Process a specific YourBench task.")
    # Add positional argument for task name
    parser.add_argument("task_name", nargs="?", help="Name of the task to process")
    # Keep the original --task-name as an optional argument
    parser.add_argument(
        "--task-name", dest="task_name_opt", help="Name of the task to process"
    )
    args = parser.parse_args()

    # Use either the positional or optional argument
    task_name = args.task_name or args.task_name_opt
    if not task_name:
        parser.error(
            "Task name must be provided either as a positional argument or with --task-name"
        )

    # Import these functions only when needed at runtime
    from yourbench.utils.load_task_config import get_available_tasks, load_task_config

    available_tasks = get_available_tasks()
    if task_name not in available_tasks:
        logger.error(
            "Invalid task: {}. Available tasks: {}",
            task_name,
            ", ".join(available_tasks),
        )
        sys.exit(1)

    logger.info("Loading configuration for task: {}", task_name)
    config = load_task_config(task_name)
    print(config)

    logger.info("Beginning pipeline processing...", task=task_name)
    process_pipeline(config)
    logger.info("Pipeline processing completed for task: {}", task_name)


if __name__ == "__main__":
    main()
