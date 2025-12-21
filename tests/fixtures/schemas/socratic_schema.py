"""Socratic method question schema.

Designed for dialectic learning and guided discovery.
"""

from typing import Literal
from pydantic import BaseModel, Field


class SocraticQuestion(BaseModel):
    """Question schema for Socratic dialogue and guided learning."""

    dialectic_goal: str = Field(description="What realization or insight this question sequence leads toward.")
    question: str = Field(description="An open-ended question that prompts reflection, not a fact query.")
    probing_follow_ups: list[str] = Field(
        min_length=2, max_length=4, description="2-4 follow-up questions to deepen understanding."
    )
    expected_student_reasoning: str = Field(description="The line of reasoning we expect students to develop.")
    common_misconceptions: list[str] = Field(
        default_factory=list, description="Misconceptions this question helps surface."
    )
    scaffolding_hints: list[str] = Field(default_factory=list, description="Hints to provide if student is stuck.")
    depth_level: Literal["surface", "conceptual", "relational", "extended-abstract"] = Field(
        description="SOLO taxonomy level targeted."
    )
    citations: list[str] = Field(description="Text passages that inform the dialogue.")
