"""Unit tests for schema prompt generator."""

from typing import Literal

from pydantic import Field, BaseModel

from yourbench.utils.question_schemas import OpenEndedQuestion, MultiChoiceQuestion
from yourbench.utils.schema_prompt_generator import (
    _get_type_description,
    generate_example_json,
    generate_schema_instructions,
)


class TestGenerateSchemaInstructions:
    """Tests for generate_schema_instructions function."""

    def test_generates_field_list(self):
        instructions = generate_schema_instructions(OpenEndedQuestion)
        assert "thought_process" in instructions
        assert "question_type" in instructions
        assert "question" in instructions
        assert "answer" in instructions
        assert "estimated_difficulty" in instructions
        assert "citations" in instructions

    def test_includes_field_descriptions(self):
        instructions = generate_schema_instructions(OpenEndedQuestion)
        assert "Explain why" in instructions  # From thought_process description

    def test_includes_type_info(self):
        instructions = generate_schema_instructions(OpenEndedQuestion)
        assert "string" in instructions
        assert "integer" in instructions

    def test_includes_constraints(self):
        instructions = generate_schema_instructions(OpenEndedQuestion)
        # estimated_difficulty has ge=1, le=10
        assert "min: 1" in instructions
        assert "max: 10" in instructions

    def test_includes_literal_values(self):
        instructions = generate_schema_instructions(OpenEndedQuestion)
        assert "analytical" in instructions
        assert "factual" in instructions

    def test_includes_docstring(self):
        instructions = generate_schema_instructions(OpenEndedQuestion)
        # OpenEndedQuestion has a docstring
        assert "open-ended questions" in instructions.lower() or "output_json" in instructions.lower()

    def test_multi_choice_includes_choices(self):
        instructions = generate_schema_instructions(MultiChoiceQuestion)
        assert "choices" in instructions
        assert "min items: 4" in instructions
        assert "max items: 4" in instructions


class TestGetTypeDescription:
    """Tests for _get_type_description function."""

    def test_str_type(self):
        assert _get_type_description(str) == "string"

    def test_int_type(self):
        assert _get_type_description(int) == "integer"

    def test_list_type(self):
        assert "array" in _get_type_description(list[str])

    def test_literal_type(self):
        result = _get_type_description(Literal["a", "b", "c"])
        assert "one of" in result
        assert '"a"' in result
        assert '"b"' in result
        assert '"c"' in result


class TestGenerateExampleJson:
    """Tests for generate_example_json function."""

    def test_generates_valid_json(self):
        import json

        result = generate_example_json(OpenEndedQuestion)
        # Should be parseable JSON
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert "question" in parsed[0]

    def test_includes_all_fields(self):
        import json

        result = generate_example_json(OpenEndedQuestion)
        parsed = json.loads(result)
        example = parsed[0]
        assert "thought_process" in example
        assert "question_type" in example
        assert "estimated_difficulty" in example


class TestCustomSchema:
    """Tests with custom schemas."""

    def test_custom_schema_with_field_descriptions(self):
        class TechnicalQuestion(BaseModel):
            """Technical documentation Q&A."""

            reasoning: str = Field(description="Why this tests understanding")
            question: str = Field(description="The question text")
            difficulty: Literal["easy", "medium", "hard"] = Field(description="Skill level")

        instructions = generate_schema_instructions(TechnicalQuestion)
        assert "reasoning" in instructions
        assert "difficulty" in instructions
        assert "easy" in instructions
        assert "medium" in instructions
        assert "hard" in instructions
