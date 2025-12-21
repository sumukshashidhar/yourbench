"""Research paper question schema.

Designed for academic papers, scientific studies, and research methodology.
"""

from typing import Literal

from pydantic import Field, BaseModel


class ResearchQuestion(BaseModel):
    """Question schema for academic and scientific research papers."""

    analytical_rationale: str = Field(
        description="Explain how this question tests understanding of the research methodology or findings."
    )
    question_focus: Literal[
        "methodology",
        "hypothesis",
        "results",
        "limitations",
        "implications",
        "literature-review",
        "statistical-analysis",
        "reproducibility",
    ] = Field(description="The aspect of research this question examines.")
    question: str = Field(description="A question about the research. Be precise about methods, data, or conclusions.")
    answer: str = Field(description="Evidence-based answer grounded in the paper's content.")
    confidence_level: Literal["established", "supported", "tentative", "speculative"] = Field(
        description="How strongly the answer is supported by the source material."
    )
    key_statistics: list[str] = Field(
        default_factory=list, description="Relevant statistics, p-values, or quantitative findings."
    )
    methodological_notes: str | None = Field(
        default=None, description="Notes about the methodology relevant to the answer."
    )
    citations: list[str] = Field(description="Exact quotes from the research paper.")
