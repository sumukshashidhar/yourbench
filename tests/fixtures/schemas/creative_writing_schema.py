"""Creative writing and literary analysis schema.

Designed for fiction, poetry, and literary criticism.
"""

from typing import Literal

from pydantic import Field, BaseModel


class LiteraryQuestion(BaseModel):
    """Question schema for creative writing and literary analysis."""

    critical_lens: str = Field(description="The interpretive approach used to formulate this question.")
    question_focus: Literal[
        "theme",
        "character",
        "symbolism",
        "narrative-structure",
        "style",
        "historical-context",
        "intertextuality",
        "reader-response",
    ] = Field(description="The literary element being examined.")
    question: str = Field(description="A literary analysis question. Encourage close reading and interpretation.")
    interpretive_answer: str = Field(description="A nuanced answer that supports claims with textual evidence.")
    alternative_readings: list[str] = Field(
        default_factory=list, description="Other valid interpretations of the text."
    )
    literary_devices: list[str] = Field(
        default_factory=list, description="Literary devices relevant to this question (metaphor, irony, etc.)."
    )
    engagement_level: Literal["surface", "analytical", "critical", "theoretical"] = Field(
        description="The depth of literary engagement required."
    )
    textual_evidence: list[str] = Field(description="Exact quotes from the text supporting the interpretation.")
