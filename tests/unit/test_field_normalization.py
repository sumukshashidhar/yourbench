"""Tests for field alias normalization in parsing engine."""

from yourbench.utils.parsing_engine import (
    FIELD_ALIASES,
    DIFFICULTY_MAPPINGS,
    _normalize_pair_fields,
)


class TestNormalizePairFields:
    """Tests for the _normalize_pair_fields function."""

    def test_reasoning_to_thought_process(self):
        """Test that 'reasoning' is mapped to 'thought_process'."""
        pair = {"reasoning": "My reasoning here", "question": "What is X?"}
        result = _normalize_pair_fields(pair)
        assert "thought_process" in result
        assert result["thought_process"] == "My reasoning here"
        assert "reasoning" not in result

    def test_explanation_to_thought_process(self):
        """Test that 'explanation' is mapped to 'thought_process'."""
        pair = {"explanation": "My explanation", "question": "What is Y?"}
        result = _normalize_pair_fields(pair)
        assert "thought_process" in result
        assert result["thought_process"] == "My explanation"
        assert "explanation" not in result

    def test_does_not_overwrite_existing(self):
        """Test that alias does not overwrite existing thought_process."""
        pair = {"thought_process": "Original", "reasoning": "Should be ignored"}
        result = _normalize_pair_fields(pair)
        assert result["thought_process"] == "Original"
        assert result["reasoning"] == "Should be ignored"

    def test_difficulty_string_to_int(self):
        """Test that string difficulty values are converted to integers."""
        for string_val, expected_int in DIFFICULTY_MAPPINGS.items():
            pair = {"difficulty": string_val}
            result = _normalize_pair_fields(pair)
            assert result["estimated_difficulty"] == expected_int

    def test_difficulty_case_insensitive(self):
        """Test that difficulty conversion is case insensitive."""
        pair = {"difficulty": "BEGINNER"}
        result = _normalize_pair_fields(pair)
        assert result["estimated_difficulty"] == 2

    def test_unknown_difficulty_defaults_to_5(self):
        """Test that unknown difficulty strings default to 5."""
        pair = {"difficulty": "unknown_level"}
        result = _normalize_pair_fields(pair)
        assert result["estimated_difficulty"] == 5

    def test_numeric_difficulty_unchanged(self):
        """Test that numeric difficulty values are not changed."""
        pair = {"difficulty": 7}
        result = _normalize_pair_fields(pair)
        # 'difficulty' is aliased to 'estimated_difficulty'
        assert result["estimated_difficulty"] == 7

    def test_no_changes_to_standard_fields(self):
        """Test that standard fields are not modified."""
        pair = {
            "thought_process": "Original",
            "question": "What?",
            "answer": "Something",
            "estimated_difficulty": 5,
        }
        result = _normalize_pair_fields(pair)
        assert result == pair

    def test_multiple_aliases(self):
        """Test that multiple aliases can be applied."""
        pair = {
            "reasoning": "My reasoning",
            "difficulty": "advanced",
            "question": "What?",
        }
        result = _normalize_pair_fields(pair)
        assert result["thought_process"] == "My reasoning"
        assert result["estimated_difficulty"] == 7
        assert "reasoning" not in result
        assert "difficulty" not in result

    def test_all_field_aliases_defined(self):
        """Test that all expected field aliases are defined."""
        expected_aliases = [
            "reasoning",
            "explanation",
            "rationale",
            "thinking",
            "difficulty",
            "complexity",
        ]
        for alias in expected_aliases:
            assert alias in FIELD_ALIASES, f"Missing alias: {alias}"
