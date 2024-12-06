from typing import List, Literal

from pydantic import BaseModel, Field


class QuestionAnswerPair(BaseModel):
    """A single question and answer pair"""
    question_type: Literal["analytical", "boolean", "factual"] = Field(
        description="The type of question"
    )
    question: str = Field(description="The question")
    answer: str = Field(description="The answer to the question")
    estimated_difficulty: int = Field(description="The estimated difficulty of the question")
    citations: List[str] = Field(description="The citations for the answer. These should be verbatim quotes from the text chunk.")
