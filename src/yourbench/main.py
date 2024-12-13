import argparse

# from yourbench.preprocessing.create_chunks import create_chunks_for_documents
# from yourbench.preprocessing.create_multihop_chunks import create_multihop_chunks
from yourbench.preprocessing.dataset_generation import generate_dataset

# from yourbench.preprocessing.generate_summaries import generate_summaries_for_documents
# from yourbench.question_generation.generate_questions import (
#     generate_multihop_questions,
#     generate_single_shot_questions,
# )
from yourbench.utils.load_task_config import get_available_tasks, load_task_config


def _process_generate_dataset(config: dict):
    # check
    if config["selected_choices"]["generate_dataset"]["execute"]:
        generate_dataset(config=config)
    else:
        print("Skipping dataset generation as it is not specified in the task config")


def process_pipeline(config: dict):
    """Process the yourbench pipeline for a given task"""
    # process dataset generation
    _process_generate_dataset(config)

    # if "pipeline_config" not in config:
    #     raise ValueError("Pipeline config not found in task config")

    # if "generate_dataset" in config["pipeline_config"]:
    #     generate_dataset(config=config)
    # else:
    #     print("Skipping dataset generation as it is not specified in the task config")

    # # check if we need summary generation
    # if "generate_summaries" in config["pipeline_config"]:
    #     generate_summaries_for_documents(
    #         config["datasets"]["document_dataset_name"], config
    #     )
    # else:
    #     print("Skipping summary generation as it is not specified in the task config")

    # if "create_chunks" in config["pipeline_config"]:
    #     create_chunks_for_documents(
    #         config["datasets"]["document_dataset_name"], config
    #     )
    # else:
    #     print("Skipping chunk creation as it is not specified in the task config")

    # if "make_multihop_chunks" in config["pipeline_config"]:
    #     create_multihop_chunks(config)
    # else:
    #     print("Skipping multihop chunk creation as it is not specified in the task config")

    # if "create_single_shot_questions" in config["pipeline_config"]:
    #     generate_single_shot_questions(
    #         config["datasets"]["document_dataset_name"], config
    #     )
    # else:
    #     print("Skipping single-shot question creation as it is not specified in the task config")

    # if "create_multihop_questions" in config["pipeline_config"]:
    #     generate_multihop_questions(config)
    # else:
    #     print("Skipping multihop question creation as it is not specified in the task config")


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
