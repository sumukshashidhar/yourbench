import random
from typing import List, Literal
from dataclasses import dataclass

import yaml
from loguru import logger
from rich.table import Table
from rich.console import Console
from datasets import exceptions as ds_exceptions

from yourbench.utils.dataset_engine import custom_load_dataset, ConfigurationError
from yourbench.utils.loading_engine import load_config


@dataclass
class Question:
    question: str
    answer: str
    question_type: str
    choices: List[str]
    difficulty: str
    index: int

    @classmethod
    def from_dataset_row(cls, row: dict, index: int) -> "Question":
        return cls(
            question=row.get("question", ""),
            answer=row.get("self_answer", ""),
            question_type=row.get("self_assessed_question_type", "unknown"),
            choices=row.get("choices", []) or [],
            difficulty=str(row.get("estimated_difficulty", "")),
            index=index,
        )

    @property
    def choices_display(self) -> str:
        return "\n".join(self.choices) if self.choices else "N/A"


class QuestionDisplay:
    def __init__(self, console: Console):
        self.console = console

    def create_table(self) -> Table:
        table = Table(show_header=True, header_style="bold cyan", show_lines=True)
        table.add_column("Q #", style="dim", width=5)
        table.add_column("Q Type", style="white", width=16)
        table.add_column("Question", style="white", no_wrap=False)
        table.add_column("Answer", style="white", no_wrap=False)
        table.add_column("Choices", style="white", no_wrap=False)
        table.add_column("Difficulty", style="white", justify="center", width=10)
        return table

    def display_questions(self, questions: List[Question], title: str, title_style: str) -> None:
        if not questions:
            self.console.print(f"[bold red]No {title.lower()} found or it's empty.[/bold red]")
            return

        self.console.print(f"[{title_style}]=== {title} ===[/{title_style}]\n")
        table = self.create_table()

        for idx, question in enumerate(questions, 1):
            table.add_row(
                str(idx),
                question.question_type,
                question.question,
                question.answer,
                question.choices_display,
                question.difficulty,
            )

        self.console.print(table)
        self.console.print()


class QuestionLoader:
    def __init__(self, config: dict, sample_size: int):
        self.config = config
        self.sample_size = sample_size

    def load_questions(self, subset: Literal["single_shot_questions", "multi_hop_questions"]) -> List[Question]:
        dataset = custom_load_dataset(config=self.config, subset=subset)
        if not dataset:
            return []

        indices = random.sample(range(len(dataset)), min(self.sample_size, len(dataset)))
        return [Question.from_dataset_row(dataset[i], i) for i in indices]


def run(*cli_args: List[str]) -> None:
    """
    Usage:
        yourbench analyze view_sample_questions path/to/config.yaml [sample_size]

    This command loads up to 'sample_size' questions from both
    'single_shot_questions' and 'multi_hop_questions' subsets, then prints
    them in a Rich table showing relevant details:
      - Question Type
      - Actual Question
      - Answer (or multiple-choice correct letter)
      - Choices (if any)
      - Difficulty
      - Citations (if any)

    Args:
        *cli_args: The CLI arguments passed after 'view_sample_questions'
                   e.g. ["my_config.yaml", "5"]
    """
    if not cli_args:
        logger.error("No arguments provided. Usage: yourbench analyze view_sample_questions CONFIG_PATH [SAMPLE_SIZE]")
        return

    config_path = cli_args[0]
    sample_size = int(cli_args[1]) if len(cli_args) > 1 and cli_args[1].isdigit() else 5

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at '{config_path}'. Aborting.")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML from '{config_path}': {e}")
        return

    loader = QuestionLoader(config, sample_size)
    display = QuestionDisplay(Console())

    # Display single-shot questions
    try:
        single_shot_questions = loader.load_questions("single_shot_questions")
        display.display_questions(single_shot_questions, "Single-Shot Questions (Detailed)", "bold magenta")
    except (ConfigurationError, ValueError, ds_exceptions.DatasetNotFoundError, KeyError) as e:
        logger.error(f"Failed to load single-shot questions: {e}")

    # Display multi-hop questions
    try:
        multi_hop_questions = loader.load_questions("multi_hop_questions")
        display.display_questions(multi_hop_questions, "Multi-Hop Questions (Detailed)", "bold green")
    except (ConfigurationError, ValueError, ds_exceptions.DatasetNotFoundError, KeyError) as e:
        logger.error(f"Failed to load multi-hop questions: {e}")

    # display rewritten questions
    try:
        rewritten_questions = loader.load_questions("single_shot_questions_rewritten")
        display.display_questions(rewritten_questions, "Rewritten Questions (Detailed)", "bold yellow")
    except (ConfigurationError, ValueError, ds_exceptions.DatasetNotFoundError, KeyError) as e:
        logger.error(f"Failed to load rewritten questions: {e}")

    try:
        rewritten_questions = loader.load_questions("multi_hop_questions_rewritten")
        display.display_questions(rewritten_questions, "Rewritten Questions (Detailed)", "bold yellow")
    except (ConfigurationError, ValueError, ds_exceptions.DatasetNotFoundError, KeyError) as e:
        logger.error(f"Failed to load rewritten questions: {e}")
