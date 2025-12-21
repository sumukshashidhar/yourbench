"""Legal document question schema.

Designed for contracts, regulations, case law, and legal documents.
"""

from typing import Literal

from pydantic import Field, BaseModel


class LegalQuestion(BaseModel):
    """Question schema for legal documents and regulations."""

    legal_reasoning: str = Field(description="Explain the legal significance of this question.")
    legal_domain: Literal[
        "contract",
        "regulatory",
        "constitutional",
        "criminal",
        "civil",
        "intellectual-property",
        "employment",
        "corporate",
    ] = Field(description="The area of law this question addresses.")
    question: str = Field(description="Legal question requiring precise interpretation.")
    answer: str = Field(description="Answer with specific legal reasoning and references.")
    key_provisions: list[str] = Field(description="Specific provisions, clauses, or sections relevant to the answer.")
    potential_ambiguities: list[str] = Field(
        default_factory=list, description="Areas where the legal text might be interpreted differently."
    )
    jurisdiction_notes: str | None = Field(default=None, description="Notes about jurisdictional applicability.")
    citations: list[str] = Field(description="Exact quotes from the legal document.")
