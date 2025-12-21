"""Medical and clinical question schema.

Designed for medical literature, clinical guidelines, and healthcare content.
"""

from typing import Literal

from pydantic import Field, BaseModel


class MedicalQuestion(BaseModel):
    """Question schema for medical and clinical documentation."""

    clinical_reasoning: str = Field(description="Explain the clinical relevance of this question.")
    question_domain: Literal[
        "diagnosis",
        "treatment",
        "pathophysiology",
        "pharmacology",
        "prevention",
        "prognosis",
        "epidemiology",
        "ethics",
    ] = Field(description="The medical domain this question covers.")
    question: str = Field(
        description="A clinically relevant question. Be specific about conditions, treatments, or mechanisms."
    )
    answer: str = Field(description="Evidence-based answer with clinical context.")
    evidence_level: Literal["guideline", "meta-analysis", "rct", "cohort", "case-series", "expert-opinion"] = Field(
        description="The level of evidence supporting this answer."
    )
    clinical_pearls: list[str] = Field(default_factory=list, description="Key clinical takeaways or mnemonics.")
    contraindications: list[str] = Field(
        default_factory=list, description="Important contraindications or warnings to note."
    )
    difficulty: Literal["student", "resident", "attending", "specialist"] = Field(
        description="Target training level for this question."
    )
    citations: list[str] = Field(description="Exact quotes from the source material.")
