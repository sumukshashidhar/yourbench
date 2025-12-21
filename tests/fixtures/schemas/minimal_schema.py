"""Minimal question schema.

Tests that the system works with very simple schemas.
"""

from pydantic import Field, BaseModel


class MinimalQuestion(BaseModel):
    """Absolute minimal question schema - just question, answer, and citations."""

    question: str = Field(description="The question.")
    answer: str = Field(description="The answer.")
    citations: list[str] = Field(description="Supporting quotes.")
