"""Educational content question schema.

Designed for textbooks, learning materials, and instructional content.
"""

from typing import Literal
from pydantic import BaseModel, Field


class EducationalQuestion(BaseModel):
    """Question schema for educational and instructional content."""

    learning_objective: str = Field(description="What specific learning outcome this question tests.")
    blooms_level: Literal[
        "remember",
        "understand",
        "apply",
        "analyze",
        "evaluate",
        "create",
    ] = Field(description="Bloom's taxonomy level this question targets.")
    question: str = Field(description="Clear, pedagogically sound question.")
    answer: str = Field(description="Complete answer suitable for a student.")
    common_misconceptions: list[str] = Field(
        default_factory=list, description="Misconceptions a student might have about this topic."
    )
    scaffolding_hints: list[str] = Field(default_factory=list, description="Hints to help a struggling student.")
    grade_level: Literal["elementary", "middle-school", "high-school", "undergraduate", "graduate"] = Field(
        description="Target educational level."
    )
    citations: list[str] = Field(description="Quotes from the source material.")
