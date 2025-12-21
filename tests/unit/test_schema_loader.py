"""Unit tests for schema loader."""

import tempfile

import pytest

from yourbench.utils.schema_loader import SchemaLoadError, load_schema_from_spec
from yourbench.utils.question_schemas import OpenEndedQuestion, MultiChoiceQuestion


class TestLoadSchemaFromSpec:
    """Tests for load_schema_from_spec function."""

    def test_none_spec_returns_default_open_ended(self):
        schema = load_schema_from_spec(None, "open-ended")
        assert schema == OpenEndedQuestion

    def test_none_spec_returns_default_multi_choice(self):
        schema = load_schema_from_spec(None, "multi-choice")
        assert schema == MultiChoiceQuestion

    def test_invalid_spec_format_raises(self):
        with pytest.raises(SchemaLoadError, match="Expected format"):
            load_schema_from_spec("no_colon_here.py", "open-ended")

    def test_missing_file_raises(self):
        with pytest.raises(SchemaLoadError, match="not found"):
            load_schema_from_spec("/nonexistent/path.py:MyClass", "open-ended")

    def test_non_python_file_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not python")
            with pytest.raises(SchemaLoadError, match="must be a Python file"):
                load_schema_from_spec(f"{f.name}:MyClass", "open-ended")

    def test_missing_class_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("from pydantic import BaseModel\nclass OtherClass(BaseModel): pass")
            f.flush()
            with pytest.raises(SchemaLoadError, match="not found"):
                load_schema_from_spec(f"{f.name}:MissingClass", "open-ended")

    def test_non_basemodel_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("class NotPydantic: pass")
            f.flush()
            with pytest.raises(SchemaLoadError, match="must be a Pydantic BaseModel"):
                load_schema_from_spec(f"{f.name}:NotPydantic", "open-ended")

    def test_valid_custom_schema_loads(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(
                "from pydantic import BaseModel\n"
                "class CustomQuestion(BaseModel):\n"
                "    question: str\n"
                "    answer: str\n"
            )
            f.flush()
            schema = load_schema_from_spec(f"{f.name}:CustomQuestion", "open-ended")
            assert schema.__name__ == "CustomQuestion"
            # Verify we can instantiate it
            instance = schema(question="Q?", answer="A")
            assert instance.question == "Q?"
