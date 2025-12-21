"""Unit tests for question schema modules."""

import pytest
from pydantic import ValidationError

from yourbench.utils.question_schemas import (
    OpenEndedQuestion,
    MultiChoiceQuestion,
    get_default_schema,
)


class TestOpenEndedQuestion:
    """Tests for OpenEndedQuestion schema."""

    def test_valid_open_ended_question(self):
        q = OpenEndedQuestion(
            thought_process="Tests understanding of core concept",
            question_type="analytical",
            question="What is the main idea?",
            answer="The main idea is...",
            estimated_difficulty=5,
            citations=["quote from text"],
        )
        assert q.question_type == "analytical"
        assert q.estimated_difficulty == 5

    def test_difficulty_range_validation(self):
        with pytest.raises(ValidationError):
            OpenEndedQuestion(
                thought_process="Test",
                question_type="factual",
                question="Q?",
                answer="A",
                estimated_difficulty=11,  # Invalid: > 10
                citations=[],
            )

    def test_invalid_question_type(self):
        with pytest.raises(ValidationError):
            OpenEndedQuestion(
                thought_process="Test",
                question_type="invalid_type",
                question="Q?",
                answer="A",
                estimated_difficulty=5,
                citations=[],
            )

    def test_all_question_types_valid(self):
        """Verify all documented question types are accepted."""
        valid_types = [
            "analytical",
            "application-based",
            "clarification",
            "counterfactual",
            "conceptual",
            "true-false",
            "factual",
            "open-ended",
            "false-premise",
            "edge-case",
        ]
        for qt in valid_types:
            q = OpenEndedQuestion(
                thought_process="Test",
                question_type=qt,
                question="Q?",
                answer="A",
                estimated_difficulty=5,
                citations=["cite"],
            )
            assert q.question_type == qt


class TestMultiChoiceQuestion:
    """Tests for MultiChoiceQuestion schema."""

    def test_valid_multi_choice_question(self):
        q = MultiChoiceQuestion(
            thought_process="Tests recall",
            question_type="factual",
            question="Which option is correct?",
            choices=["(A) Option 1", "(B) Option 2", "(C) Option 3", "(D) Option 4"],
            answer="A",
            estimated_difficulty=3,
            citations=["source text"],
        )
        assert q.answer == "A"
        assert len(q.choices) == 4

    def test_exactly_four_choices_required(self):
        with pytest.raises(ValidationError):
            MultiChoiceQuestion(
                thought_process="Test",
                question_type="factual",
                question="Q?",
                choices=["(A) Only", "(B) Three", "(C) Choices"],  # Only 3
                answer="A",
                estimated_difficulty=5,
                citations=[],
            )

    def test_answer_must_be_letter(self):
        with pytest.raises(ValidationError):
            MultiChoiceQuestion(
                thought_process="Test",
                question_type="factual",
                question="Q?",
                choices=["(A) 1", "(B) 2", "(C) 3", "(D) 4"],
                answer="E",  # Invalid letter
                estimated_difficulty=5,
                citations=[],
            )


class TestGetDefaultSchema:
    """Tests for get_default_schema function."""

    def test_open_ended_mode(self):
        schema = get_default_schema("open-ended")
        assert schema == OpenEndedQuestion

    def test_multi_choice_mode(self):
        schema = get_default_schema("multi-choice")
        assert schema == MultiChoiceQuestion

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown question mode"):
            get_default_schema("invalid-mode")

    def test_mode_case_insensitive(self):
        assert get_default_schema("OPEN-ENDED") == OpenEndedQuestion
        assert get_default_schema("Multi-Choice") == MultiChoiceQuestion
