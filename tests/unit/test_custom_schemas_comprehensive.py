"""Comprehensive tests for custom Pydantic question schemas.

Tests schema loading, prompt generation, and field normalization
for a variety of custom schema configurations.
"""

from pathlib import Path

import pytest

from yourbench.utils.schema_loader import load_schema_from_spec
from yourbench.utils.question_schemas import get_default_schema
from yourbench.utils.schema_prompt_generator import (
    generate_example_json,
    generate_critical_reminders,
    generate_schema_instructions,
)


SCHEMAS_DIR = Path(__file__).parent.parent / "fixtures" / "schemas"


class TestSchemaLoading:
    """Tests for loading custom schemas from file paths."""

    @pytest.mark.parametrize(
        "schema_file,class_name",
        [
            ("technical_schema.py", "TechnicalQuestion"),
            ("philosophical_schema.py", "PhilosophicalQuestion"),
            ("research_schema.py", "ResearchQuestion"),
            ("educational_schema.py", "EducationalQuestion"),
            ("legal_schema.py", "LegalQuestion"),
            ("mcq_schema.py", "CustomMCQ"),
            ("minimal_schema.py", "MinimalQuestion"),
            ("medical_schema.py", "MedicalQuestion"),
            ("nested_schema.py", "NestedQuestion"),
            ("creative_writing_schema.py", "LiteraryQuestion"),
            ("exam_prep_schema.py", "ExamQuestion"),
            ("socratic_schema.py", "SocraticQuestion"),
        ],
    )
    def test_load_all_custom_schemas(self, schema_file: str, class_name: str):
        """Verify all custom schemas can be loaded."""
        spec = f"{SCHEMAS_DIR}/{schema_file}:{class_name}"
        schema = load_schema_from_spec(spec, question_mode="open-ended")
        assert schema is not None
        assert schema.__name__ == class_name

    def test_load_schema_with_absolute_path(self):
        """Test loading schema with absolute path."""
        abs_path = Path.cwd() / SCHEMAS_DIR / "minimal_schema.py"
        spec = f"{abs_path}:MinimalQuestion"
        schema = load_schema_from_spec(spec, question_mode="open-ended")
        assert schema.__name__ == "MinimalQuestion"

    def test_default_schema_fallback(self):
        """When spec is None, returns default schema."""
        schema = load_schema_from_spec(None, question_mode="open-ended")
        assert schema.__name__ == "OpenEndedQuestion"

        schema = load_schema_from_spec(None, question_mode="multi-choice")
        assert schema.__name__ == "MultiChoiceQuestion"


class TestPromptGeneration:
    """Tests for generating prompt instructions from schemas."""

    @pytest.mark.parametrize(
        "schema_file,class_name,expected_fields",
        [
            (
                "technical_schema.py",
                "TechnicalQuestion",
                ["reasoning", "question_type", "difficulty", "prerequisites"],
            ),
            (
                "philosophical_schema.py",
                "PhilosophicalQuestion",
                ["contemplation", "question_nature", "possible_perspectives"],
            ),
            ("medical_schema.py", "MedicalQuestion", ["clinical_reasoning", "evidence_level", "contraindications"]),
            (
                "exam_prep_schema.py",
                "ExamQuestion",
                ["bloom_taxonomy_level", "marking_scheme", "time_allocation_minutes"],
            ),
            ("socratic_schema.py", "SocraticQuestion", ["dialectic_goal", "probing_follow_ups", "scaffolding_hints"]),
        ],
    )
    def test_schema_instructions_contain_expected_fields(
        self, schema_file: str, class_name: str, expected_fields: list[str]
    ):
        """Verify generated instructions mention all expected fields."""
        spec = f"{SCHEMAS_DIR}/{schema_file}:{class_name}"
        schema = load_schema_from_spec(spec, question_mode="open-ended")
        instructions = generate_schema_instructions(schema)

        for field in expected_fields:
            assert field in instructions, f"Expected field '{field}' not found in instructions"

    def test_nested_schema_generates_valid_instructions(self):
        """Test that nested schemas produce comprehensible instructions."""
        spec = f"{SCHEMAS_DIR}/nested_schema.py:NestedQuestion"
        schema = load_schema_from_spec(spec, question_mode="open-ended")
        instructions = generate_schema_instructions(schema)

        assert "structured_answer" in instructions
        assert "primary_citations" in instructions

    def test_critical_reminders_reflect_schema_fields(self):
        """Critical reminders should reference actual schema field names."""
        spec = f"{SCHEMAS_DIR}/technical_schema.py:TechnicalQuestion"
        schema = load_schema_from_spec(spec, question_mode="open-ended")
        reminders = generate_critical_reminders(schema)

        assert "reasoning" in reminders.lower()
        assert "thought_process" not in reminders.lower()

    def test_example_output_is_valid_json_structure(self):
        """Example output should be valid JSON-like structure."""
        spec = f"{SCHEMAS_DIR}/minimal_schema.py:MinimalQuestion"
        schema = load_schema_from_spec(spec, question_mode="open-ended")
        example = generate_example_json(schema)

        assert "question" in example
        assert "answer" in example
        assert "citations" in example


class TestDefaultSchemas:
    """Tests for the default open-ended and multi-choice schemas."""

    def test_open_ended_default_has_required_fields(self):
        schema = get_default_schema("open-ended")
        fields = set(schema.model_fields.keys())
        expected = {"thought_process", "question_type", "question", "answer", "estimated_difficulty", "citations"}
        assert expected.issubset(fields)

    def test_multi_choice_default_has_choices_field(self):
        schema = get_default_schema("multi-choice")
        fields = set(schema.model_fields.keys())
        assert "choices" in fields
        assert "answer" in fields

    def test_default_schema_instructions_match_hardcoded_prompt(self):
        """Ensure default schema instructions are compatible with existing prompts."""
        schema = get_default_schema("open-ended")
        instructions = generate_schema_instructions(schema)

        assert "thought_process" in instructions
        assert "estimated_difficulty" in instructions
        assert "citations" in instructions


class TestEdgeCases:
    """Edge case tests for schema handling."""

    def test_schema_with_list_constraints(self):
        """Test schemas that have min/max_length on lists."""
        spec = f"{SCHEMAS_DIR}/philosophical_schema.py:PhilosophicalQuestion"
        schema = load_schema_from_spec(spec, question_mode="open-ended")

        instructions = generate_schema_instructions(schema)
        assert "possible_perspectives" in instructions

    def test_schema_with_integer_constraints(self):
        """Test schemas that have ge/le constraints on integers."""
        spec = f"{SCHEMAS_DIR}/exam_prep_schema.py:ExamQuestion"
        schema = load_schema_from_spec(spec, question_mode="open-ended")

        instructions = generate_schema_instructions(schema)
        assert "time_allocation_minutes" in instructions
        assert "marks" in instructions

    def test_schema_with_optional_fields(self):
        """Test schemas with Optional fields."""
        spec = f"{SCHEMAS_DIR}/technical_schema.py:TechnicalQuestion"
        schema = load_schema_from_spec(spec, question_mode="open-ended")

        instructions = generate_schema_instructions(schema)
        assert "code_snippet" in instructions

    def test_schema_with_pattern_constraint(self):
        """Test schema with regex pattern on field."""
        schema = get_default_schema("multi-choice")
        instructions = generate_schema_instructions(schema)
        assert "answer" in instructions

    def test_mcq_schema_with_dict_field(self):
        """Test MCQ schema that has a dict field (distractor_explanations)."""
        spec = f"{SCHEMAS_DIR}/mcq_schema.py:CustomMCQ"
        schema = load_schema_from_spec(spec, question_mode="multi-choice")

        instructions = generate_schema_instructions(schema)
        assert "distractor_explanations" in instructions
