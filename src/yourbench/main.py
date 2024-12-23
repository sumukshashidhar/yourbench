import argparse

from yourbench.preprocessing.create_chunks import create_chunks_for_documents
from yourbench.preprocessing.create_multihop_chunks import create_multihop_chunks
from yourbench.preprocessing.dataset_generation import generate_dataset
from yourbench.preprocessing.generate_summaries import generate_summaries_for_documents
from yourbench.postprocessing.reformat_dataset_for_judge import reformat_for_judging
from yourbench.question_answering.answer_questions import answer_questions_with_llm
from yourbench.judge.judge_answers import judge_answers
from yourbench.question_generation.generate_questions import (
    generate_multihop_questions,
    generate_single_shot_questions,
)
from yourbench.utils.load_task_config import get_available_tasks, load_task_config


def _process_generate_dataset(config: dict):
    # check
    if config["selected_choices"]["generate_dataset"]["execute"]:
        generate_dataset(config=config)
    else:
        print("Skipping dataset generation as it is not specified in the task config")


def _process_generate_summaries(config: dict):

    if config["selected_choices"]["generate_summaries"]["execute"]:
        generate_summaries_for_documents(config=config)
    else:
        print("Skipping summary generation as it is not specified in the task config")


def _process_create_chunks(config: dict):
    if config["selected_choices"]["create_chunks"]["execute"]:
        create_chunks_for_documents(config=config)
    else:
        print("Skipping chunk creation as it is not specified in the task config")


def _process_make_multihop_chunks(config: dict):
    if config["selected_choices"]["make_multihop_chunks"]["execute"]:
        create_multihop_chunks(config=config)
    else:
        print("Skipping multihop chunk creation as it is not specified in the task config")


def _process_create_single_shot_questions(config: dict):
    if config["selected_choices"]["create_single_shot_questions"]["execute"]:
        generate_single_shot_questions(config=config)
    else:
        print("Skipping single-shot question creation as it is not specified in the task config")


def _process_create_multihop_questions(config: dict):
    if config["selected_choices"]["create_multihop_questions"]["execute"]:
        generate_multihop_questions(config=config)
    else:
        print("Skipping multihop question creation as it is not specified in the task config")


def _answer_questions_with_llm(config: dict):
    if config["selected_choices"]["answer_questions_with_llm"]["execute"]:
        answer_questions_with_llm(config=config)
    else:
        print("Skipping question answering as it is not specified in the task config")

def _process_reformat_for_judging(config: dict):
    if config["selected_choices"]["reformat_for_judging"]["execute"]:
        reformat_for_judging(config=config)
    else:
        print("Skipping question answering as it is not specified in the task config")


def _process_judge_answers(config: dict):
    if config["selected_choices"]["judge"]["execute"]:
        judge_answers(config=config)
    else:
        print("Skipping question answering as it is not specified in the task config")


def process_pipeline(config: dict):
    """Process the yourbench pipeline for a given task"""
    # process dataset generation
    _process_generate_dataset(config)
    _process_generate_summaries(config)
    _process_create_chunks(config)
    _process_make_multihop_chunks(config)
    _process_create_single_shot_questions(config)
    _process_create_multihop_questions(config)
    _answer_questions_with_llm(config)
    _process_reformat_for_judging(config)
    _process_judge_answers(config)


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
