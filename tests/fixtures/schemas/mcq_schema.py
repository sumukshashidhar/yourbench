"""Custom multiple-choice question schema.

Demonstrates how to create MCQ-specific schemas with custom difficulty.
"""

from typing import Literal
from pydantic import BaseModel, Field


class CustomMCQ(BaseModel):
    """Enhanced multiple-choice question with detailed distractors."""

    distractor_rationale: str = Field(description="Explain why the wrong answers are plausible but incorrect.")
    question_type: Literal[
        "factual-recall",
        "concept-application",
        "inference",
        "analysis",
        "evaluation",
    ] = Field(description="The cognitive skill being tested.")
    question: str = Field(description="The question stem. Must be complete and unambiguous.")
    choices: list[str] = Field(
        min_length=4, max_length=4, description="Exactly 4 choices: (A), (B), (C), (D). Only one is correct."
    )
    correct_answer: Literal["A", "B", "C", "D"] = Field(description="The letter of the correct answer.")
    distractor_explanations: dict[str, str] = Field(
        description="Map of wrong answer letters to explanations of why they're wrong."
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Question difficulty level.")
    citations: list[str] = Field(description="Source text supporting the correct answer.")
