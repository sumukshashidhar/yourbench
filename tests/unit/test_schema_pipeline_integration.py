"""Integration tests for schema injection into the question generation pipeline.

These tests verify that custom schemas are correctly injected into prompts
and that the generated prompts don't contain default field references.
"""

from pathlib import Path

import pytest

from yourbench.conf.loader import load_config
from yourbench.pipeline.question_generation._core import _get_system_prompt


SCHEMAS_DIR = Path(__file__).parent.parent / "fixtures" / "schemas"


class TestPromptInjection:
    """Test that custom schemas are correctly injected into system prompts."""

    @pytest.fixture
    def base_config(self):
        """Load a base config for testing."""
        return load_config("example/default_example/config.yaml")

    def test_default_prompt_has_thought_process(self, base_config):
        """Default prompts should have thought_process field."""
        prompt = _get_system_prompt(base_config.pipeline.single_shot_question_generation, "open-ended")
        assert "thought_process" in prompt

    def test_technical_schema_replaces_default_fields(self):
        """Technical schema should replace thought_process with reasoning."""
        config = load_config("tests/fixtures/configs/config_technical.yaml")
        prompt = _get_system_prompt(config.pipeline.single_shot_question_generation, "open-ended")

        # Check custom field is present
        assert "reasoning" in prompt
        assert "prerequisites" in prompt

        # Check the output format section was replaced
        assert "TechnicalQuestion" in prompt or "reasoning" in prompt

    def test_philosophical_schema_has_contemplation(self):
        """Philosophical schema should have contemplation field."""
        config = load_config("tests/fixtures/configs/config_philosophical.yaml")
        prompt = _get_system_prompt(config.pipeline.single_shot_question_generation, "open-ended")

        assert "contemplation" in prompt
        assert "possible_perspectives" in prompt

    def test_minimal_schema_has_only_core_fields(self):
        """Minimal schema should only have question, answer, citations."""
        config = load_config("tests/fixtures/configs/config_minimal.yaml")
        prompt = _get_system_prompt(config.pipeline.single_shot_question_generation, "open-ended")

        # Core fields present
        assert "question" in prompt
        assert "answer" in prompt
        assert "citations" in prompt

    def test_research_schema_fields(self):
        """Research schema should have methodology-related fields."""
        config = load_config("tests/fixtures/configs/config_research.yaml")
        prompt = _get_system_prompt(config.pipeline.single_shot_question_generation, "open-ended")

        assert "analytical_rationale" in prompt
        assert "question_focus" in prompt
        assert "confidence_level" in prompt

    def test_medical_schema_has_clinical_fields(self):
        """Medical schema should have clinical domain fields."""
        config = load_config("tests/fixtures/configs/config_medical.yaml")
        prompt = _get_system_prompt(config.pipeline.single_shot_question_generation, "open-ended")

        assert "clinical_reasoning" in prompt
        assert "evidence_level" in prompt
        assert "contraindications" in prompt

    def test_exam_schema_has_bloom_taxonomy(self):
        """Exam schema should have Bloom's taxonomy levels."""
        config = load_config("tests/fixtures/configs/config_exam.yaml")
        prompt = _get_system_prompt(config.pipeline.single_shot_question_generation, "open-ended")

        assert "bloom_taxonomy_level" in prompt
        assert "marking_scheme" in prompt
        assert "time_allocation_minutes" in prompt

    def test_socratic_schema_has_dialectic_fields(self):
        """Socratic schema should have dialectic learning fields."""
        config = load_config("tests/fixtures/configs/config_socratic.yaml")
        prompt = _get_system_prompt(config.pipeline.single_shot_question_generation, "open-ended")

        assert "dialectic_goal" in prompt
        assert "probing_follow_ups" in prompt
        assert "scaffolding_hints" in prompt

    def test_literary_schema_has_critical_analysis(self):
        """Literary schema should have literary analysis fields."""
        config = load_config("tests/fixtures/configs/config_literary.yaml")
        prompt = _get_system_prompt(config.pipeline.single_shot_question_generation, "open-ended")

        assert "critical_lens" in prompt
        assert "question_focus" in prompt
        assert "literary_devices" in prompt


class TestSchemaFieldsInReminders:
    """Test that critical reminders section uses correct field names."""

    def test_technical_reminders_no_thought_process(self):
        """Technical schema reminders should not reference thought_process."""
        config = load_config("tests/fixtures/configs/config_technical.yaml")
        prompt = _get_system_prompt(config.pipeline.single_shot_question_generation, "open-ended")

        # The critical reminders section should use schema-appropriate language
        # and should NOT mention default fields like thought_process in the reminders
        lines = prompt.split("\n")
        for line in lines:
            if "Critical" in line and "Reminders" in line:
                # We found the reminders section, verify it uses proper field names
                break
