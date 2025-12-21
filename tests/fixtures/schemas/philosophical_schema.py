"""Philosophical and conceptual question schema.

Designed for philosophy texts, ethics discussions, and abstract concepts.
"""

from typing import Literal

from pydantic import Field, BaseModel


class PhilosophicalQuestion(BaseModel):
    """Question schema for philosophical and conceptual content."""

    contemplation: str = Field(
        description="Reflect on why this question probes deep understanding of the philosophical concept."
    )
    question_nature: Literal[
        "ontological",
        "epistemological",
        "ethical",
        "metaphysical",
        "logical",
        "phenomenological",
        "dialectical",
        "hermeneutic",
    ] = Field(description="The branch of philosophy this question engages with.")
    question: str = Field(description="The philosophical question. Encourage deep reflection and nuanced thinking.")
    possible_perspectives: list[str] = Field(
        min_length=2, max_length=4, description="2-4 distinct philosophical perspectives one might take in answering."
    )
    synthesis: str = Field(description="A synthesized answer that acknowledges multiple perspectives.")
    related_thinkers: list[str] = Field(
        default_factory=list, description="Philosophers or thinkers whose work relates to this question."
    )
    conceptual_difficulty: Literal["accessible", "moderate", "complex", "profound"] = Field(
        description="How conceptually demanding this question is."
    )
    citations: list[str] = Field(description="Exact quotes from the source text.")
