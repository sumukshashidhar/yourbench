from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional

class ChunkAnalysis(BaseModel):
    """Analysis of a single chunk and its relationships to others"""
    chunk_id: str
    content_summary: str
    relationships: List[str] = Field(..., description="How this chunk relates to others")

class GeneratedIntegrativeQAPair(BaseModel):
    """Generated question-answer pair requiring integration across chunks"""
    document_analysis: str = Field(..., description="Analysis of the document based on summary")
    chunks_analysis: List[ChunkAnalysis] = Field(..., description="Analysis of each provided chunk")
    integration_points: List[str] = Field(..., description="Key points where information connects")
    potential_integrative_questions: List[str] = Field(..., description="Possible integrative questions")
    best_direction: str = Field(..., description="Chosen question direction and rationale")
    direct_line_quotes: Dict[str, List[str]] = Field(
        default_factory=lambda: {"dummy_chunk": ["No quotes available"]},
        description="Verbatim quotes by chunk ID"
    )
    question: str = Field(..., description="The integrative question")
    answer: str = Field(..., description="Answer showing synthesis")
    reasoning: str = Field(..., description="How information is integrated")
    chunks_used: List[str] = Field(..., description="Chunk IDs used in answer")
    kind: str = Field(..., description="Question type (e.g., integrative_factual)")
    estimated_difficulty: int = Field(..., ge=1, le=5, description="Difficulty level 1-5")

    @validator('estimated_difficulty')
    def validate_difficulty(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Difficulty must be between 1 and 5')
        return v

    @validator('chunks_used')
    def validate_chunks_used(cls, v):
        if not v:
            raise ValueError('Must use at least one chunk')
        return v

    @validator('direct_line_quotes')
    def validate_direct_line_quotes(cls, v):
        if not v:
            return {"dummy_chunk": ["No quotes available"]}
        return v