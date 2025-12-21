"""Technical documentation question schema.

Designed for API docs, code tutorials, and technical specifications.
"""

from typing import Literal

from pydantic import Field, BaseModel


class TechnicalQuestion(BaseModel):
    """Question schema for technical documentation and API references."""

    reasoning: str = Field(
        description="Explain why this question effectively tests understanding of the technical concept."
    )
    question_type: Literal[
        "api-usage",
        "debugging",
        "implementation",
        "architecture",
        "performance",
        "best-practice",
        "error-handling",
        "integration",
    ] = Field(description="The category of technical question.")
    question: str = Field(description="The technical question. Be specific about APIs, methods, or concepts.")
    answer: str = Field(description="Complete technical answer with code examples if applicable.")
    difficulty: Literal["beginner", "intermediate", "advanced", "expert"] = Field(
        description="Target skill level for the question."
    )
    prerequisites: list[str] = Field(
        default_factory=list, description="Concepts the reader should understand before attempting this question."
    )
    code_snippet: str | None = Field(default=None, description="Relevant code example demonstrating the answer.")
    citations: list[str] = Field(description="Exact quotes from the source documentation.")
