"""Exam preparation question schema.

Designed for standardized test prep and exam-style questions.
"""

from typing import Literal
from pydantic import BaseModel, Field


class ExamQuestion(BaseModel):
    """Question schema optimized for exam preparation."""

    pedagogical_purpose: str = Field(description="What skill or knowledge this question tests.")
    bloom_taxonomy_level: Literal[
        "remember",
        "understand",
        "apply",
        "analyze",
        "evaluate",
        "create",
    ] = Field(description="The cognitive level per Bloom's taxonomy.")
    question: str = Field(description="The exam question. Be clear and unambiguous.")
    model_answer: str = Field(description="A model answer that would receive full marks.")
    marking_scheme: list[str] = Field(description="Key points that should appear in a correct answer.")
    common_mistakes: list[str] = Field(default_factory=list, description="Common errors students make on this topic.")
    time_allocation_minutes: int = Field(
        ge=1, le=60, description="Suggested time to spend on this question in minutes."
    )
    marks: int = Field(ge=1, le=20, description="Number of marks this question is worth.")
    citations: list[str] = Field(description="Source material supporting the question.")
