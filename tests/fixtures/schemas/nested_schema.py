"""Schema with nested types and complex structures.

Tests that the system handles complex Pydantic models correctly.
"""

from typing import Literal

from pydantic import Field, BaseModel


class Citation(BaseModel):
    """A structured citation reference."""

    text: str = Field(description="The exact quoted text.")
    page: int | None = Field(default=None, description="Page number if available.")
    section: str | None = Field(default=None, description="Section name if available.")


class Answer(BaseModel):
    """Structured answer with confidence."""

    main_answer: str = Field(description="The primary answer.")
    supporting_details: list[str] = Field(default_factory=list, description="Additional supporting points.")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence in the answer.")


class NestedQuestion(BaseModel):
    """Question schema with nested structures to test complex types."""

    reasoning: str = Field(description="Why this question tests understanding.")
    question: str = Field(description="The question text.")
    structured_answer: Answer = Field(description="The structured answer object.")
    primary_citations: list[Citation] = Field(description="Primary source citations with metadata.")
    related_topics: list[str] = Field(default_factory=list, description="Related topics for further study.")
    difficulty: int = Field(ge=1, le=5, description="Difficulty on a 1-5 scale.")
