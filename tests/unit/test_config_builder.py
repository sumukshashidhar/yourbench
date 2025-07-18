"""Tests for the config_builder module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from yourbench.config_builder import (
    load_config,
    save_config,
    write_env_file,
    create_yourbench_config,
    validate_api_key_format,
)
from yourbench.utils.configuration_engine import (
    ModelConfig,
    YourbenchConfig,
    HuggingFaceConfig,
)


class TestAPIKeyValidation:
    """Test API key validation functionality."""

    def test_valid_env_var_format(self):
        """Test that environment variable format is valid."""
        valid, msg = validate_api_key_format("$HF_TOKEN")
        assert valid
        assert msg == "$HF_TOKEN"

    def test_empty_key_valid(self):
        """Test that empty key is valid."""
        valid, msg = validate_api_key_format("")
        assert valid
        assert msg == ""

    def test_invalid_real_key_format(self):
        """Test that real API keys are flagged as invalid."""
        valid, msg = validate_api_key_format("sk-1234567890abcdef")
        assert not valid
        assert "environment variable format" in msg

    def test_invalid_hf_token_format(self):
        """Test that Hugging Face tokens are flagged as invalid."""
        valid, msg = validate_api_key_format("hf_1234567890abcdef")
        assert not valid
        assert "environment variable format" in msg

    def test_short_key_valid(self):
        """Test that short keys are considered valid."""
        valid, msg = validate_api_key_format("short")
        assert valid
        assert msg == "short"

    def test_normal_string_valid(self):
        """Test that normal strings are valid."""
        valid, msg = validate_api_key_format("some_string")
        assert valid
        assert msg == "some_string"


class TestEnvFileWriting:
    """Test .env file writing functionality."""

    def test_write_env_file_new_file(self):
        """Test writing to a new .env file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"

            # Change to temporary directory
            old_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                api_keys = {
                    "HF_TOKEN": "hf_...",
                    "OPENAI_API_KEY": "sk-...",
                }

                write_env_file(api_keys)

                # Check that file was created
                assert env_path.exists()

                # Check contents
                content = env_path.read_text()
                assert "HF_TOKEN=hf_..." in content
                assert "OPENAI_API_KEY=sk-..." in content
                assert "# API Keys added by YourBench" in content

            finally:
                os.chdir(old_cwd)

    def test_write_env_file_existing_file(self):
        """Test writing to an existing .env file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"

            # Create existing .env file
            env_path.write_text("EXISTING_VAR=existing_value\n")

            # Change to temporary directory
            old_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                api_keys = {
                    "HF_TOKEN": "hf_...",
                    "EXISTING_VAR": "should_not_overwrite",
                }

                write_env_file(api_keys)

                content = env_path.read_text()
                # Should not overwrite existing variable
                assert "EXISTING_VAR=existing_value" in content
                # Should add new variable
                assert "HF_TOKEN=hf_..." in content
                # Should not have the value that would overwrite
                assert "EXISTING_VAR=should_not_overwrite" not in content

            finally:
                os.chdir(old_cwd)

    def test_write_env_file_no_new_keys(self):
        """Test that nothing is written when no new keys are provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"

            # Create existing .env file
            original_content = "HF_TOKEN=existing_token\n"
            env_path.write_text(original_content)

            # Change to temporary directory
            old_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                api_keys = {
                    "HF_TOKEN": "new_token",  # Already exists
                }

                write_env_file(api_keys)

                # Content should be unchanged
                content = env_path.read_text()
                assert content == original_content

            finally:
                os.chdir(old_cwd)

    def test_write_env_file_permission_error(self):
        """Test handling of permission errors."""
        with patch("yourbench.config_builder.Path.open") as mock_open:
            mock_open.side_effect = PermissionError("Permission denied")

            api_keys = {"HF_TOKEN": "hf_..."}

            # Should not raise exception
            write_env_file(api_keys)


class TestConfigSaveLoad:
    """Test configuration saving and loading."""

    def test_save_config_basic(self):
        """Test basic configuration saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            # Create a simple config
            config = YourbenchConfig(
                hf_configuration=HuggingFaceConfig(hf_dataset_name="test_dataset"),
                model_list=[ModelConfig(model_name="test_model")],
            )

            save_config(config, config_path)

            # Check that file was created
            assert config_path.exists()

            # Check contents
            with config_path.open() as f:
                saved_data = yaml.safe_load(f)

            assert saved_data["hf_configuration"]["hf_dataset_name"] == "test_dataset"
            assert len(saved_data["model_list"]) == 1
            assert saved_data["model_list"][0]["model_name"] == "test_model"

    def test_save_config_creates_directory(self):
        """Test that save_config creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "subdir" / "test_config.yaml"

            config = YourbenchConfig(
                hf_configuration=HuggingFaceConfig(hf_dataset_name="test_dataset"),
            )

            save_config(config, config_path)

            # Check that directory and file were created
            assert config_path.exists()
            assert config_path.parent.exists()

    def test_save_config_none_value_cleanup(self):
        """Test that None values are cleaned up in saved config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            config = YourbenchConfig(
                hf_configuration=HuggingFaceConfig(hf_dataset_name="test_dataset"),
                model_list=[ModelConfig(model_name="test_model", base_url=None)],
            )

            save_config(config, config_path)

            # Check that None values are removed
            with config_path.open() as f:
                saved_data = yaml.safe_load(f)

            # base_url should not be present since it was None
            assert "base_url" not in saved_data["model_list"][0]

    def test_load_config(self):
        """Test configuration loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            # Create a YAML config file
            config_data = {
                "hf_configuration": {
                    "hf_dataset_name": "test_dataset",
                    "private": True,
                },
                "model_list": [
                    {
                        "model_name": "test_model",
                        "api_key": "$HF_TOKEN",
                    }
                ],
                "pipeline": {
                    "ingestion": {"run": True},
                    "summarization": {"run": True},
                },
            }

            with config_path.open("w") as f:
                yaml.dump(config_data, f)

            # Load the config
            loaded_config = load_config(config_path)

            assert isinstance(loaded_config, YourbenchConfig)
            assert loaded_config.hf_configuration.hf_dataset_name == "test_dataset"
            assert loaded_config.hf_configuration.private
            assert len(loaded_config.model_list) == 1
            assert loaded_config.model_list[0].model_name == "test_model"
            assert loaded_config.pipeline_config.ingestion.run
            assert loaded_config.pipeline_config.summarization.run


class TestConfigCreation:
    """Test configuration creation functionality."""

    @patch("yourbench.config_builder.console")
    @patch("yourbench.config_builder.get_random_name")
    def test_create_yourbench_config_simple(self, mock_get_name, mock_console):
        """Test creating a simple YourBench configuration."""
        mock_get_name.return_value = "test_dataset"

        config = create_yourbench_config(simple=True)

        assert isinstance(config, YourbenchConfig)
        assert config.hf_configuration.hf_dataset_name == "test_dataset"
        assert len(config.model_list) == 1
        assert config.model_list[0].model_name == "Qwen/Qwen3-30B-A3B"
        assert config.model_list[0].provider == "fireworks-ai"
        assert config.pipeline_config.ingestion.run
        assert config.pipeline_config.summarization.run
        assert config.pipeline_config.chunking.run

    @patch("yourbench.config_builder.console")
    @patch("yourbench.config_builder.get_random_name")
    @patch("yourbench.config_builder.Confirm.ask")
    @patch("yourbench.config_builder.Prompt.ask")
    def test_create_yourbench_config_advanced_minimal(self, mock_prompt, mock_confirm, mock_get_name, mock_console):
        """Test creating an advanced configuration with minimal user input."""
        mock_get_name.return_value = "test_dataset"
        mock_prompt.ask.return_value = "test_dataset"
        mock_confirm.ask.return_value = False  # Always decline optional configuration

        config = create_yourbench_config(simple=False)

        assert isinstance(config, YourbenchConfig)
        assert config.hf_configuration.hf_dataset_name == "test_dataset"

    def test_create_yourbench_config_integration(self):
        """Integration test for configuration creation."""
        # Test that the function can be called without mocking everything
        # This mainly tests that imports work and basic structure is correct

        # Use simple mode to avoid interactive prompts
        config = create_yourbench_config(simple=True)

        # Basic structure checks
        assert isinstance(config, YourbenchConfig)
        assert isinstance(config.hf_configuration, HuggingFaceConfig)
        assert isinstance(config.model_list, list)
        assert len(config.model_list) > 0
        assert isinstance(config.model_list[0], ModelConfig)


class TestConfigurationSystem:
    """Integration tests for the entire configuration system."""

    def test_full_config_cycle(self):
        """Test complete configuration creation, saving, and loading cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "full_test.yaml"

            # Create config
            original_config = create_yourbench_config(simple=True)

            # Save config
            save_config(original_config, config_path)

            # Load config
            loaded_config = load_config(config_path)

            # Verify they match
            assert loaded_config.hf_configuration.hf_dataset_name == original_config.hf_configuration.hf_dataset_name
            assert len(loaded_config.model_list) == len(original_config.model_list)
            assert loaded_config.model_list[0].model_name == original_config.model_list[0].model_name
            assert loaded_config.pipeline_config.ingestion.run == original_config.pipeline_config.ingestion.run

    def test_environment_variable_handling(self):
        """Test that environment variables are properly handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "env_test.yaml"

            # Create config with environment variables
            config_data = {
                "hf_configuration": {
                    "hf_dataset_name": "test_dataset",
                    "hf_token": "$HF_TOKEN",
                    "hf_organization": "$HF_ORGANIZATION",
                },
                "model_list": [
                    {
                        "model_name": "test_model",
                        "api_key": "$HF_TOKEN",
                    }
                ],
                "pipeline": {
                    "ingestion": {"run": True},
                },
            }

            with config_path.open("w") as f:
                yaml.dump(config_data, f)

            # Load config with environment variables set
            with patch.dict(os.environ, {"HF_TOKEN": "test_token", "HF_ORGANIZATION": "test_org"}):
                loaded_config = load_config(config_path)

                # Environment variables should be expanded
                assert loaded_config.hf_configuration.hf_token == "test_token"
                assert loaded_config.hf_configuration.hf_organization == "test_org"
                assert loaded_config.model_list[0].api_key == "test_token"

    def test_config_validation_errors(self):
        """Test that configuration validation catches errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid_test.yaml"

            # Create invalid config (missing required fields)
            config_data = {
                "hf_configuration": {
                    # Missing hf_dataset_name
                },
                "model_list": [
                    {
                        # Missing model_name
                        "api_key": "$HF_TOKEN",
                    }
                ],
            }

            with config_path.open("w") as f:
                yaml.dump(config_data, f)

            # Loading should work (dataclass fills in defaults)
            # But the config might not be functionally valid
            loaded_config = load_config(config_path)

            # Check that defaults are filled
            assert loaded_config.hf_configuration.hf_dataset_name is not None  # Gets random name
            assert loaded_config.model_list[0].model_name is None  # No default for this