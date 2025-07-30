"""Edge case tests for the configuration system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml
import pytest

from yourbench.config_builder import (
    save_config,
    write_env_file,
    create_model_config,
    validate_api_key_format,
)
from yourbench.utils.configuration_engine import (
    ModelConfig,
    YourbenchConfig,
    HuggingFaceConfig,
    _expand_env,
)


class TestAPIKeyValidationEdgeCases:
    """Test edge cases for API key validation."""

    def test_api_key_with_special_characters(self):
        """Test API keys with various special characters."""
        # Environment variable format should always be valid
        valid, msg = validate_api_key_format("$API_KEY_WITH_UNDERSCORES")
        assert valid

        valid, msg = validate_api_key_format("$API-KEY-WITH-DASHES")
        assert valid

        valid, msg = validate_api_key_format("$API123KEY")
        assert valid

    def test_api_key_edge_cases(self):
        """Test various edge cases for API key validation."""
        # Exactly 10 characters (boundary case)
        valid, msg = validate_api_key_format("1234567890")
        assert valid

        # 11 characters with suspicious pattern
        valid, msg = validate_api_key_format("sk-12345678")
        assert not valid

        # Long string without suspicious patterns
        valid, msg = validate_api_key_format("very_long_string_without_patterns")
        assert valid

        # Empty string
        valid, msg = validate_api_key_format("")
        assert valid

    def test_api_key_suspicious_patterns(self):
        """Test detection of suspicious API key patterns."""
        suspicious_keys = [
            "sk-1234567890abcdef",
            "key-abcdef1234567890",
            "api-1234567890abcdef",
            "hf_1234567890abcdef",
        ]

        for key in suspicious_keys:
            valid, msg = validate_api_key_format(key)
            assert not valid
            assert "environment variable format" in msg


class TestEnvironmentVariableEdgeCases:
    """Test edge cases for environment variable handling."""

    def test_expand_env_with_empty_env_var(self):
        """Test expansion of environment variable that exists but is empty."""
        with patch.dict(os.environ, {"EMPTY_VAR": ""}):
            result = _expand_env("$EMPTY_VAR")
            assert result == ""

    def test_expand_env_with_whitespace(self):
        """Test expansion with whitespace in environment variables."""
        with patch.dict(os.environ, {"SPACE_VAR": "  value with spaces  "}):
            result = _expand_env("$SPACE_VAR")
            assert result == "  value with spaces  "

    def test_expand_env_nested_variables(self):
        """Test that nested variable expansion doesn't occur."""
        with patch.dict(os.environ, {"NESTED": "$OTHER_VAR", "OTHER_VAR": "value"}):
            result = _expand_env("$NESTED")
            # Should return the literal value, not expand further
            assert result == "$OTHER_VAR"

    def test_hf_organization_fallback_edge_cases(self):
        """Test HF_ORGANIZATION fallback edge cases."""
        # Test when HF_TOKEN is empty
        with patch.dict(os.environ, {"HF_TOKEN": ""}):
            result = _expand_env("$HF_ORGANIZATION")
            assert result == "$HF_ORGANIZATION"

        # Test when whoami returns unexpected format
        with patch.dict(os.environ, {"HF_TOKEN": "hf_token"}):
            with patch("yourbench.utils.configuration_engine.whoami") as mock_whoami:
                mock_whoami.return_value = {}  # Missing 'name' key
                result = _expand_env("$HF_ORGANIZATION")
                # Should handle KeyError gracefully
                assert result == "$HF_ORGANIZATION"


class TestConfigurationEdgeCases:
    """Test edge cases in configuration handling."""

    def test_model_config_with_none_values(self):
        """Test ModelConfig with None values."""
        config = ModelConfig(
            model_name=None,
            base_url=None,
            api_key=None,
            provider=None,
        )

        # Should handle None values gracefully
        assert config.model_name is None
        assert config.base_url is None
        assert config.api_key is None
        assert config.provider is None  # No auto-assignment when explicitly None

    def test_model_config_auto_provider_logic(self):
        """Test auto provider assignment logic."""
        # With base_url, no auto provider
        config1 = ModelConfig(base_url="https://api.example.com")
        assert config1.provider is None

        # Without base_url, should get auto provider
        config2 = ModelConfig()
        assert config2.provider == "auto"

        # With explicit provider, keep it
        config3 = ModelConfig(provider="custom")
        assert config3.provider == "custom"

    def test_hf_config_with_path_objects(self):
        """Test HuggingFaceConfig with Path objects."""
        config = HuggingFaceConfig(
            local_dataset_dir=Path("/custom/path"),
        )

        assert isinstance(config.local_dataset_dir, Path)
        assert str(config.local_dataset_dir) == "/custom/path"

    def test_yourbench_config_empty_model_list(self):
        """Test YourbenchConfig with empty model list."""
        config = YourbenchConfig(model_list=[])

        # Should not crash and should have empty model roles
        assert config.model_list == []
        assert len(config.model_roles) == 0

    def test_yourbench_config_model_without_name(self):
        """Test YourbenchConfig with model that has no name."""
        model = ModelConfig(model_name=None)
        config = YourbenchConfig(model_list=[model])

        # Should handle gracefully
        assert len(config.model_list) == 1
        # No model roles should be assigned since model has no name
        assert len(config.model_roles) == 0


class TestYAMLHandlingEdgeCases:
    """Test edge cases in YAML handling."""

    def test_yaml_with_null_values(self):
        """Test YAML loading with null values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "null_config.yaml"

            config_data = {
                "hf_configuration": {
                    "hf_dataset_name": "test",
                    "hf_organization": None,
                    "private": None,
                },
                "model_list": [
                    {
                        "model_name": "test_model",
                        "base_url": None,
                        "api_key": None,
                    }
                ],
                "pipeline": None,
            }

            with config_path.open("w") as f:
                yaml.dump(config_data, f)

            from yourbench.utils.configuration_engine import YourbenchConfig

            config = YourbenchConfig.from_yaml(config_path)

            # Should handle null values gracefully
            assert config.hf_configuration.hf_dataset_name == "test"
            assert len(config.model_list) == 1
            assert config.model_list[0].model_name == "test_model"

    def test_yaml_with_empty_sections(self):
        """Test YAML loading with empty sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "empty_config.yaml"

            config_data = {
                "hf_configuration": {},
                "model_list": [],
                "pipeline": {},
            }

            with config_path.open("w") as f:
                yaml.dump(config_data, f)

            from yourbench.utils.configuration_engine import YourbenchConfig

            config = YourbenchConfig.from_yaml(config_path)

            # Should handle empty sections gracefully
            assert isinstance(config.hf_configuration, HuggingFaceConfig)
            assert config.model_list == []
            assert hasattr(config.pipeline_config, "ingestion")

    def test_yaml_with_unexpected_keys(self):
        """Test YAML loading with unexpected/unknown keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "unexpected_config.yaml"

            config_data = {
                "hf_configuration": {
                    "hf_dataset_name": "test",
                    "unknown_field": "should_be_ignored",
                },
                "model_list": [
                    {
                        "model_name": "test_model",
                        "unknown_model_field": "should_be_ignored",
                    }
                ],
                "pipeline": {
                    "unknown_stage": {"run": True},
                    "ingestion": {"run": True},
                },
                "unknown_top_level": "should_be_ignored",
            }

            with config_path.open("w") as f:
                yaml.dump(config_data, f)

            from yourbench.utils.configuration_engine import YourbenchConfig

            # Should not crash, should ignore unknown fields
            config = YourbenchConfig.from_yaml(config_path)

            assert config.hf_configuration.hf_dataset_name == "test"
            assert len(config.model_list) == 1
            assert config.pipeline_config.ingestion.run


class TestErrorHandlingEdgeCases:
    """Test error handling edge cases."""

    def test_save_config_permission_error(self):
        """Test save_config handling of permission errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a read-only directory
            readonly_dir = Path(tmpdir) / "readonly"
            readonly_dir.mkdir()
            readonly_dir.chmod(0o444)  # Read-only

            config_path = readonly_dir / "config.yaml"
            config = YourbenchConfig()

            try:
                # Should handle permission error gracefully
                with pytest.raises((PermissionError, OSError)):
                    save_config(config, config_path)
            finally:
                # Clean up - restore permissions
                readonly_dir.chmod(0o755)

    def test_write_env_file_edge_cases(self):
        """Test write_env_file edge cases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                # Test with malformed existing .env file
                env_path = Path(".env")
                env_path.write_text("MALFORMED LINE WITHOUT EQUALS\nVALID=value\n")

                api_keys = {"NEW_KEY": "new_value"}

                # Should handle malformed lines gracefully
                write_env_file(api_keys)

                # Should still add new key
                content = env_path.read_text()
                assert "NEW_KEY=new_value" in content

            finally:
                os.chdir(old_cwd)

    def test_create_model_config_edge_cases(self):
        """Test create_model_config edge cases."""
        # These tests require mocking user input since the function is interactive
        with patch("yourbench.config_builder.console"):
            with patch("yourbench.config_builder.Prompt.ask") as mock_prompt:
                with patch("yourbench.config_builder.IntPrompt.ask") as mock_int_prompt:
                    with patch("yourbench.config_builder.Confirm.ask") as mock_confirm:
                        # Test edge case: choice 1 (HF) with no provider
                        mock_prompt.ask.return_value = "test_model"
                        mock_int_prompt.ask.return_value = 1
                        mock_confirm.ask.return_value = False

                        config = create_model_config([])

                        assert isinstance(config, ModelConfig)
                        assert config.model_name == "test_model"


class TestTypeHandlingEdgeCases:
    """Test type handling and conversion edge cases."""

    def test_path_conversion_edge_cases(self):
        """Test Path object conversion edge cases."""
        from yourbench.utils.configuration_engine import IngestionConfig

        # Test with empty string
        config = IngestionConfig(
            source_documents_dir="",
            output_dir="",
        )

        assert isinstance(config.source_documents_dir, Path)
        assert isinstance(config.output_dir, Path)
        assert str(config.source_documents_dir) == "."  # Empty path becomes current dir
        assert str(config.output_dir) == "."

    def test_prompt_loading_edge_cases(self):
        """Test prompt file loading edge cases."""
        from yourbench.utils.configuration_engine import IngestionConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with non-existent prompt file
            config = IngestionConfig(pdf_llm_prompt="/nonexistent/path/prompt.md")

            # Should keep the path as-is if file doesn't exist
            assert config.pdf_llm_prompt == "/nonexistent/path/prompt.md"

            # Test with empty prompt file
            empty_prompt = Path(tmpdir) / "empty.md"
            empty_prompt.write_text("")

            config = IngestionConfig(pdf_llm_prompt=str(empty_prompt))

            # Should load empty content and strip it
            assert config.pdf_llm_prompt == ""

    def test_configuration_with_unicode(self):
        """Test configuration handling with Unicode characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "unicode_config.yaml"

            config_data = {
                "hf_configuration": {
                    "hf_dataset_name": "test_with_émojis_🚀",
                },
                "model_list": [
                    {
                        "model_name": "modèle_français",
                    }
                ],
            }

            with config_path.open("w", encoding="utf-8") as f:
                yaml.dump(config_data, f, allow_unicode=True)

            from yourbench.utils.configuration_engine import YourbenchConfig

            config = YourbenchConfig.from_yaml(config_path)

            # Should handle Unicode correctly
            assert config.hf_configuration.hf_dataset_name == "test_with_émojis_🚀"
            assert config.model_list[0].model_name == "modèle_français"


class TestConcurrencyAndStateEdgeCases:
    """Test edge cases related to concurrency and state management."""

    def test_model_config_concurrent_requests_edge_cases(self):
        """Test edge cases for concurrent request settings."""
        # Test with very low values
        config = ModelConfig(max_concurrent_requests=1)
        assert config.max_concurrent_requests == 1

        # Test with very high values
        config = ModelConfig(max_concurrent_requests=1000)
        assert config.max_concurrent_requests == 1000

        # Test with zero (might be used to disable concurrency)
        config = ModelConfig(max_concurrent_requests=0)
        assert config.max_concurrent_requests == 0

    def test_pipeline_config_stage_interdependencies(self):
        """Test edge cases in pipeline stage configuration."""
        from yourbench.utils.configuration_engine import PipelineConfig

        # All stages disabled
        config = PipelineConfig()
        assert all(
            not getattr(config, stage).run
            for stage in [
                "ingestion",
                "summarization",
                "chunking",
                "single_shot_question_generation",
                "multi_hop_question_generation",
            ]
        )

        # Test with mixed enabled/disabled stages
        config.ingestion.run = True
        config.chunking.run = True

        assert config.ingestion.run
        assert not config.summarization.run
        assert config.chunking.run
