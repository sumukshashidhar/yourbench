import argparse

from yourbench.preprocessing.generate_summaries import generate_summaries_for_documents
from yourbench.utils.load_task_config import get_available_tasks, load_task_config


def process_pipeline(config: dict):
    """Process the yourbench pipeline for a given task"""
    if "pipeline_config" not in config:
        raise ValueError("Pipeline config not found in task config")

    # check if we need summary generation
    if "generate_summaries" in config["pipeline_config"]:
        generate_summaries_for_documents(
            config["datasets"]["document_dataset_name"], config
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a specific task")
    parser.add_argument("--task-name", help="Name of the task to process")
    args = parser.parse_args()

    available_tasks = get_available_tasks()
    if args.task_name not in available_tasks:
        print(
            f"Error: '{args.task_name}' is not a valid task. Available tasks: {', '.join(available_tasks)}"
        )
        exit(1)

    config = load_task_config(args.task_name)
    process_pipeline(config)
