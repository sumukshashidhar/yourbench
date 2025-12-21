"""Consolidated tests for the custom question schema system.

Covers: schema loading, prompt generation, field normalization, and default schemas.
"""

import json
import tempfile
from typing import Literal

import pytest
from pydantic import Field, BaseModel, ValidationError

from yourbench.utils.schema_loader import SchemaLoadError, load_schema_from_spec
from yourbench.utils.parsing_engine import (
    FIELD_ALIASES,
    DIFFICULTY_MAPPINGS,
    _normalize_pair_fields,
)
from yourbench.utils.question_schemas import (
    OpenEndedQuestion,
    MultiChoiceQuestion,
    get_default_schema,
)
from yourbench.utils.schema_prompt_generator import (
    _get_type_description,
    generate_example_json,
    generate_schema_instructions,
)


class TestDefaultSchemas:
    """Tests for OpenEndedQuestion and MultiChoiceQuestion defaults."""

    def test_open_ended_valid(self):
        q = OpenEndedQuestion(
            thought_process="Tests core concept",
            question_type="analytical",
            question="What is X?",
            answer="X is...",
            estimated_difficulty=5,
            citations=["source"],
        )
        assert q.question_type == "analytical"

    def test_open_ended_difficulty_bounds(self):
        with pytest.raises(ValidationError):
            OpenEndedQuestion(
                thought_process="T",
                question_type="factual",
                question="Q?",
                answer="A",
                estimated_difficulty=11,
                citations=[],
            )

    def test_open_ended_invalid_type(self):
        with pytest.raises(ValidationError):
            OpenEndedQuestion(
                thought_process="T",
                question_type="invalid",
                question="Q?",
                answer="A",
                estimated_difficulty=5,
                citations=[],
            )

    def test_multi_choice_valid(self):
        q = MultiChoiceQuestion(
            thought_process="T",
            question_type="factual",
            question="Which?",
            choices=["(A) 1", "(B) 2", "(C) 3", "(D) 4"],
            answer="A",
            estimated_difficulty=3,
            citations=["s"],
        )
        assert q.answer == "A"

    def test_multi_choice_requires_four_choices(self):
        with pytest.raises(ValidationError):
            MultiChoiceQuestion(
                thought_process="T",
                question_type="factual",
                question="Q?",
                choices=["(A) 1", "(B) 2", "(C) 3"],
                answer="A",
                estimated_difficulty=5,
                citations=[],
            )

    def test_multi_choice_answer_letter_validation(self):
        with pytest.raises(ValidationError):
            MultiChoiceQuestion(
                thought_process="T",
                question_type="factual",
                question="Q?",
                choices=["(A) 1", "(B) 2", "(C) 3", "(D) 4"],
                answer="E",
                estimated_difficulty=5,
                citations=[],
            )

    def test_get_default_schema_modes(self):
        assert get_default_schema("open-ended") == OpenEndedQuestion
        assert get_default_schema("multi-choice") == MultiChoiceQuestion
        assert get_default_schema("OPEN-ENDED") == OpenEndedQuestion

    def test_get_default_schema_invalid(self):
        with pytest.raises(ValueError, match="Unknown question mode"):
            get_default_schema("invalid")


class TestSchemaLoader:
    """Tests for load_schema_from_spec."""

    def test_none_returns_default(self):
        assert load_schema_from_spec(None, "open-ended") == OpenEndedQuestion
        assert load_schema_from_spec(None, "multi-choice") == MultiChoiceQuestion

    def test_invalid_format_raises(self):
        with pytest.raises(SchemaLoadError, match="Expected format"):
            load_schema_from_spec("no_colon.py", "open-ended")

    def test_missing_file_raises(self):
        with pytest.raises(SchemaLoadError, match="not found"):
            load_schema_from_spec("/nonexistent/path.py:Schema", "open-ended")

    def test_non_python_raises(self):
        with pytest.raises(SchemaLoadError, match="not found"):
            load_schema_from_spec("schema.txt:Schema", "open-ended")

    def test_missing_class_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("from pydantic import BaseModel\nclass Other(BaseModel): pass\n")
            f.flush()
            with pytest.raises(SchemaLoadError, match="not found in"):
                load_schema_from_spec(f"{f.name}:Missing", "open-ended")

    def test_non_basemodel_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("class NotPydantic: pass\n")
            f.flush()
            with pytest.raises(SchemaLoadError, match="must be a Pydantic"):
                load_schema_from_spec(f"{f.name}:NotPydantic", "open-ended")

    def test_valid_custom_schema(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("from pydantic import BaseModel\nclass CustomQ(BaseModel):\n    question: str\n")
            f.flush()
            schema = load_schema_from_spec(f"{f.name}:CustomQ", "open-ended")
            assert schema.__name__ == "CustomQ"

    def test_load_complex_schema(self):
        """Test loading a schema with multiple fields and Literal types."""
        code = """from typing import Literal
from pydantic import BaseModel, Field
class TechQ(BaseModel):
    reasoning: str = Field(description="Why")
    question: str
    difficulty: Literal["easy", "hard"] = Field(description="Level")
    citations: list[str]
"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(code)
            f.flush()
            schema = load_schema_from_spec(f"{f.name}:TechQ", "open-ended")
            assert schema.__name__ == "TechQ"
            assert "reasoning" in schema.model_fields


class TestPromptGeneration:
    """Tests for schema prompt generation."""

    def test_instructions_contain_fields(self):
        instructions = generate_schema_instructions(OpenEndedQuestion)
        for field in ["thought_process", "question_type", "question", "answer", "estimated_difficulty", "citations"]:
            assert field in instructions

    def test_instructions_include_descriptions(self):
        instructions = generate_schema_instructions(OpenEndedQuestion)
        assert "Explain why" in instructions

    def test_instructions_include_constraints(self):
        instructions = generate_schema_instructions(OpenEndedQuestion)
        assert "min: 1" in instructions and "max: 10" in instructions

    def test_multi_choice_includes_choices(self):
        instructions = generate_schema_instructions(MultiChoiceQuestion)
        assert "choices" in instructions and "min items: 4" in instructions

    def test_type_descriptions(self):
        assert _get_type_description(str) == "string"
        assert _get_type_description(int) == "integer"
        assert "array" in _get_type_description(list[str])
        result = _get_type_description(Literal["a", "b"])
        assert "one of" in result and '"a"' in result

    def test_example_json_valid(self):
        result = generate_example_json(OpenEndedQuestion)
        parsed = json.loads(result)
        assert isinstance(parsed, list) and "question" in parsed[0]

    def test_custom_schema_instructions(self):
        class TechQ(BaseModel):
            reasoning: str = Field(description="Why")
            difficulty: Literal["easy", "hard"] = Field(description="Level")

        instructions = generate_schema_instructions(TechQ)
        assert "reasoning" in instructions and "easy" in instructions

    def test_critical_reminders_generation(self):
        from yourbench.utils.schema_prompt_generator import generate_critical_reminders

        reminders = generate_critical_reminders(OpenEndedQuestion)
        assert "Critical Reminders" in reminders and "citations" in reminders.lower()


class TestFieldNormalization:
    """Tests for field alias mapping."""

    def test_reasoning_to_thought_process(self):
        result = _normalize_pair_fields({"reasoning": "My reason", "question": "Q?"})
        assert result["thought_process"] == "My reason" and "reasoning" not in result

    def test_explanation_to_thought_process(self):
        result = _normalize_pair_fields({"explanation": "Explain", "question": "Q?"})
        assert result["thought_process"] == "Explain"

    def test_does_not_overwrite_existing(self):
        result = _normalize_pair_fields({"thought_process": "Original", "reasoning": "Ignored"})
        assert result["thought_process"] == "Original" and result["reasoning"] == "Ignored"

    def test_difficulty_string_to_int(self):
        for string_val, expected_int in DIFFICULTY_MAPPINGS.items():
            result = _normalize_pair_fields({"difficulty": string_val})
            assert result["estimated_difficulty"] == expected_int

    def test_difficulty_case_insensitive(self):
        result = _normalize_pair_fields({"difficulty": "BEGINNER"})
        assert result["estimated_difficulty"] == 2

    def test_unknown_difficulty_defaults_to_5(self):
        result = _normalize_pair_fields({"difficulty": "unknown"})
        assert result["estimated_difficulty"] == 5

    def test_numeric_difficulty(self):
        result = _normalize_pair_fields({"difficulty": 7})
        assert result["estimated_difficulty"] == 7

    def test_all_aliases_defined(self):
        for alias in ["reasoning", "explanation", "rationale", "thinking", "difficulty", "complexity"]:
            assert alias in FIELD_ALIASES
