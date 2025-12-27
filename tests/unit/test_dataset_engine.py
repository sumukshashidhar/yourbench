"""Tests for dataset engine functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from datasets import Dataset, DatasetDict
from yourbench.utils.dataset_engine import (
    _export_to_jsonl,
    _extract_settings,
    custom_save_dataset,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    data = {
        "question": ["What is 2+2?", "What is the capital of France?"],
        "answer": ["4", "Paris"],
        "difficulty": [1, 2],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def sample_dataset_dict(sample_dataset):
    """Create a sample DatasetDict for testing."""
    return DatasetDict({
        "train": sample_dataset,
        "test": Dataset.from_dict({
            "question": ["What is 3+3?"],
            "answer": ["6"],
            "difficulty": [1],
        }),
    })


@pytest.fixture
def mock_config_with_jsonl(temp_dir):
    """Create a mock config with JSONL export enabled."""
    return OmegaConf.create({
        "hf_configuration": {
            "hf_dataset_name": "test_dataset",
            "hf_organization": "test_org",
            "hf_token": "test_token",
            "local_dataset_dir": str(temp_dir / "datasets"),
            "export_jsonl": True,
            "jsonl_export_dir": str(temp_dir / "jsonl_export"),
            "private": False,
            "concat_if_exist": False,
        }
    })


class TestJsonlExport:
    """Test JSONL export functionality."""

    def test_export_single_dataset_to_jsonl(self, sample_dataset, temp_dir):
        """Test exporting a single Dataset to JSONL."""
        export_dir = temp_dir / "export"
        _export_to_jsonl(sample_dataset, export_dir)

        jsonl_file = export_dir / "dataset.jsonl"
        assert jsonl_file.exists()

        with open(jsonl_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2
        row1 = json.loads(lines[0])
        assert row1 == {"question": "What is 2+2?", "answer": "4", "difficulty": 1}

    def test_export_dataset_with_subset_name(self, sample_dataset, temp_dir):
        """Test exporting a Dataset with a specific subset name."""
        export_dir = temp_dir / "export"
        _export_to_jsonl(sample_dataset, export_dir, subset="train")

        jsonl_file = export_dir / "train.jsonl"
        assert jsonl_file.exists()

        with open(jsonl_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_export_dataset_dict_to_jsonl(self, sample_dataset_dict, temp_dir):
        """Test exporting a DatasetDict to multiple JSONL files."""
        export_dir = temp_dir / "export"
        _export_to_jsonl(sample_dataset_dict, export_dir)

        train_file = export_dir / "train.jsonl"
        assert train_file.exists()
        with open(train_file, "r") as f:
            train_lines = f.readlines()
        assert len(train_lines) == 2

        test_file = export_dir / "test.jsonl"
        assert test_file.exists()
        with open(test_file, "r") as f:
            test_lines = f.readlines()
        assert len(test_lines) == 1

        index_file = export_dir / "index.json"
        assert index_file.exists()
        with open(index_file, "r") as f:
            index_data = json.load(f)

        assert index_data["subsets"] == ["train", "test"]
        assert index_data["total_rows"] == 3

    def test_export_creates_directory_if_not_exists(self, sample_dataset, temp_dir):
        """Test that export creates the directory if it doesn't exist."""
        export_dir = temp_dir / "new_dir" / "nested"
        assert not export_dir.exists()

        _export_to_jsonl(sample_dataset, export_dir)

        assert export_dir.exists()
        assert (export_dir / "dataset.jsonl").exists()

    @patch("yourbench.utils.dataset_engine.load_from_disk")
    @patch("yourbench.utils.dataset_engine._safe_save")
    def test_custom_save_dataset_with_jsonl_export(
        self, mock_safe_save, mock_load_from_disk, sample_dataset, mock_config_with_jsonl
    ):
        """Test custom_save_dataset with JSONL export enabled."""
        mock_load_from_disk.side_effect = FileNotFoundError()

        with patch("yourbench.utils.dataset_engine._export_to_jsonl") as mock_export:
            custom_save_dataset(
                sample_dataset, mock_config_with_jsonl, subset="train", save_local=True, push_to_hub=False
            )

            mock_export.assert_called_once()
            args = mock_export.call_args[0]
            assert isinstance(args[0], DatasetDict)

    def test_extract_settings_includes_jsonl_config(self, mock_config_with_jsonl):
        """Test that _extract_settings properly extracts JSONL configuration."""
        settings = _extract_settings(mock_config_with_jsonl)

        assert settings.export_jsonl is True
        assert settings.jsonl_export_dir is not None

    def test_unicode_handling_in_jsonl(self, temp_dir):
        """Test that JSONL export handles Unicode characters correctly."""
        data = {"text": ["Hello 世界", "Émoji 🎉", "Ñoño"], "id": [1, 2, 3]}
        dataset = Dataset.from_dict(data)

        export_dir = temp_dir / "unicode_test"
        _export_to_jsonl(dataset, export_dir)

        jsonl_file = export_dir / "dataset.jsonl"
        with open(jsonl_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        row1 = json.loads(lines[0])
        assert row1["text"] == "Hello 世界"

        row2 = json.loads(lines[1])
        assert row2["text"] == "Émoji 🎉"

        row3 = json.loads(lines[2])
        assert row3["text"] == "Ñoño"
