from typing import List, Literal

from pydantic import BaseModel, Field


class QuestionAnswerPairWithThoughtProcess(BaseModel):
    """A single question and answer pair"""
    thought_process: str = Field(description="The thought process for the question and answer pair")
    question_type: Literal["analytical", "application-based", "clarification",
                          "counterfactual", "conceptual", "true-false",
                          "factual", "open-ended", "false-premise", "edge-case"]
    question: str = Field(description="The question")
    answer: str = Field(description="The answer to the question")
    estimated_difficulty: int = Field(description="The estimated difficulty of the question")
    citations: List[str] = Field(description="The citations for the answer. These should be verbatim quotes from the text chunk.")


class QuestionAnswerPair(BaseModel):
    """A single question and answer pair"""
    question_type: Literal["analytical", "application-based", "clarification",
                          "counterfactual", "conceptual", "true-false",
                          "factual", "open-ended", "false-premise", "edge-case"]
    question: str = Field(description="The question")
    answer: str = Field(description="The answer to the question")
    estimated_difficulty: int = Field(description="The estimated difficulty of the question")
    citations: List[str] = Field(description="The citations for the answer. These should be verbatim quotes from the text chunk.")
