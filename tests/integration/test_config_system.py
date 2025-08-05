"""Integration tests for the complete configuration system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml
import pytest

from yourbench.config_builder import load_config, save_config, create_yourbench_config
from yourbench.utils.configuration_engine import YourbenchConfig


class TestConfigurationSystemIntegration:
    """Test the complete configuration system end-to-end."""

    @patch.dict(os.environ, {"HF_TOKEN": "test_token"})
    def test_simple_config_creation_and_use(self):
        """Test creating a simple config and using it in pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            # Create simple configuration
            config = create_yourbench_config(simple=True)
            save_config(config, config_path)

            # Verify file exists and is valid YAML
            assert config_path.exists()
            with config_path.open() as f:
                yaml_data = yaml.safe_load(f)

            # Check structure
            assert "hf_configuration" in yaml_data
            assert "model_list" in yaml_data
            assert "pipeline" in yaml_data

            # Load back and verify
            with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
                loaded_config = load_config(config_path)
            assert isinstance(loaded_config, YourbenchConfig)
            assert loaded_config.model_list[0].model_name == "Qwen/Qwen3-30B-A3B"

    def test_config_with_environment_variables(self):
        """Test configuration with environment variable expansion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "env_config.yaml"

            # Create config with env vars
            config_data = {
                "hf_configuration": {
                    "hf_dataset_name": "test_dataset",
                    "hf_token": "$HF_TOKEN",
                    "hf_organization": "$HF_ORGANIZATION",
                    "private": False,
                },
                "model_list": [
                    {
                        "model_name": "test_model",
                        "api_key": "$HF_TOKEN",
                        "provider": "fireworks-ai",
                        "max_concurrent_requests": 16,
                    }
                ],
                "pipeline": {
                    "ingestion": {"run": True},
                    "summarization": {"run": True},
                    "chunking": {"run": True},
                },
            }

            with config_path.open("w") as f:
                yaml.dump(config_data, f)

            # Test with environment variables
            test_env = {
                "HF_TOKEN": "hf_test_token_12345",
                "HF_ORGANIZATION": "test_org",
            }

            with patch.dict(os.environ, test_env):
                config = load_config(config_path)

                # Verify env var expansion
                assert config.hf_configuration.hf_token == "hf_test_token_12345"
                assert config.hf_configuration.hf_organization == "test_org"
                assert config.model_list[0].api_key == "hf_test_token_12345"

    @patch.dict(os.environ, {"HF_TOKEN": "test_token"})
    def test_config_validation_and_defaults(self):
        """Test that configuration validation and defaults work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "minimal_config.yaml"

            # Create minimal config
            minimal_config = {
                "hf_configuration": {
                    "hf_dataset_name": "minimal_test",
                },
                "model_list": [
                    {
                        "model_name": "test_model",
                    }
                ],
                "pipeline": {},  # Empty pipeline should get defaults
            }

            with config_path.open("w") as f:
                yaml.dump(minimal_config, f)

            config = load_config(config_path)

            # Check defaults are applied
            assert not config.hf_configuration.private  # Default
            assert config.hf_configuration.local_saving  # Default
            assert config.model_list[0].max_concurrent_requests == 32  # Default
            assert config.model_list[0].encoding_name == "cl100k_base"  # Default
            assert not config.pipeline_config.ingestion.run  # Default when not specified

    @patch.dict(os.environ, {"HF_TOKEN": "test_token"})
    def test_config_roundtrip_preservation(self):
        """Test that config data is preserved through save/load cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "roundtrip_config.yaml"

            # Create complex configuration
            original_config = create_yourbench_config(simple=True)
            original_config.hf_configuration.private = True
            original_config.hf_configuration.concat_if_exist = True
            original_config.model_list[0].max_concurrent_requests = 24

            # Save and load
            save_config(original_config, config_path)
            with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
                loaded_config = load_config(config_path)

            # Verify preservation
            assert loaded_config.hf_configuration.private
            assert loaded_config.hf_configuration.concat_if_exist
            assert loaded_config.model_list[0].max_concurrent_requests == 24
            assert loaded_config.model_list[0].model_name == "Qwen/Qwen3-30B-A3B"

    @patch.dict(os.environ, {"HF_TOKEN": "test_token", "OPENAI_API_KEY": "test_openai_key"})
    def test_multiple_models_configuration(self):
        """Test configuration with multiple models and role assignments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "multi_model_config.yaml"

            config_data = {
                "hf_configuration": {
                    "hf_dataset_name": "multi_model_test",
                },
                "model_list": [
                    {
                        "model_name": "vision_model",
                        "provider": "fireworks-ai",
                        "api_key": "$HF_TOKEN",
                    },
                    {
                        "model_name": "text_model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "$OPENAI_API_KEY",
                    },
                ],
                "model_roles": {
                    "ingestion": ["vision_model"],
                    "summarization": ["text_model"],
                    "single_shot_question_generation": ["text_model"],
                    "multi_hop_question_generation": ["text_model"],
                },
                "pipeline": {
                    "ingestion": {"run": True},
                    "summarization": {"run": True},
                },
            }

            with config_path.open("w") as f:
                yaml.dump(config_data, f)

            config = load_config(config_path)

            # Verify multiple models
            assert len(config.model_list) == 2
            assert config.model_list[0].model_name == "vision_model"
            assert config.model_list[1].model_name == "text_model"

            # Verify role assignments
            assert config.model_roles["ingestion"] == ["vision_model"]
            assert config.model_roles["summarization"] == ["text_model"]

    @patch.dict(os.environ, {"HF_TOKEN": "test_token"})
    def test_pipeline_stage_configuration(self):
        """Test comprehensive pipeline stage configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "pipeline_config.yaml"

            config_data = {
                "hf_configuration": {
                    "hf_dataset_name": "pipeline_test",
                },
                "model_list": [
                    {
                        "model_name": "test_model",
                    }
                ],
                "pipeline": {
                    "ingestion": {
                        "run": True,
                        "source_documents_dir": "custom/source",
                        "output_dir": "custom/output",
                        "llm_ingestion": True,
                        "pdf_dpi": 600,
                    },
                    "summarization": {
                        "run": True,
                        "max_tokens": 16384,
                        "token_overlap": 256,
                    },
                    "chunking": {
                        "run": True,
                        "l_max_tokens": 1024,
                        "h_min": 3,
                        "h_max": 7,
                        "num_multihops_factor": 2,
                    },
                    "single_shot_question_generation": {
                        "run": True,
                        "question_mode": "multi-choice",
                    },
                    "multi_hop_question_generation": {
                        "run": True,
                        "question_mode": "open-ended",
                    },
                    "cross_document_question_generation": {
                        "run": True,
                        "max_combinations": 50,
                        "chunks_per_document": 2,
                        "num_docs_per_combination": [2, 5],
                        "random_seed": 123,
                    },
                    "question_rewriting": {
                        "run": True,
                        "additional_instructions": "Make questions more engaging",
                    },
                    "prepare_lighteval": {"run": True},
                    "citation_score_filtering": {"run": True},
                },
            }

            with config_path.open("w") as f:
                yaml.dump(config_data, f)

            config = load_config(config_path)

            # Verify ingestion config
            assert config.pipeline_config.ingestion.run
            assert str(config.pipeline_config.ingestion.source_documents_dir) == "custom/source"
            assert str(config.pipeline_config.ingestion.output_dir) == "custom/output"
            assert config.pipeline_config.ingestion.llm_ingestion
            assert config.pipeline_config.ingestion.pdf_dpi == 600

            # Verify summarization config
            assert config.pipeline_config.summarization.run
            assert config.pipeline_config.summarization.max_tokens == 16384
            assert config.pipeline_config.summarization.token_overlap == 256

            # Verify chunking config
            assert config.pipeline_config.chunking.run
            assert config.pipeline_config.chunking.l_max_tokens == 1024
            assert config.pipeline_config.chunking.h_min == 3
            assert config.pipeline_config.chunking.h_max == 7
            assert config.pipeline_config.chunking.num_multihops_factor == 2

            # Verify question generation configs
            assert config.pipeline_config.single_shot_question_generation.run
            assert config.pipeline_config.single_shot_question_generation.question_mode == "multi-choice"
            assert config.pipeline_config.multi_hop_question_generation.run
            assert config.pipeline_config.multi_hop_question_generation.question_mode == "open-ended"

            # Verify cross-document config
            assert config.pipeline_config.cross_document_question_generation.run
            assert config.pipeline_config.cross_document_question_generation.max_combinations == 50
            assert config.pipeline_config.cross_document_question_generation.chunks_per_document == 2
            assert config.pipeline_config.cross_document_question_generation.num_docs_per_combination == [2, 5]
            assert config.pipeline_config.cross_document_question_generation.random_seed == 123

            # Verify question rewriting config
            assert config.pipeline_config.question_rewriting.run
            assert config.pipeline_config.question_rewriting.additional_instructions == "Make questions more engaging"

    @patch.dict(os.environ, {"HF_TOKEN": "test_token"})
    def test_backward_compatibility(self):
        """Test that old configuration format still works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "legacy_config.yaml"

            # Old format with 'models' instead of 'model_list'
            legacy_config = {
                "hf_configuration": {
                    "hf_dataset_name": "legacy_test",
                    "token": "$HF_TOKEN",  # Old field name
                },
                "models": [  # Old key name
                    {
                        "model_name": "legacy_model",
                        "api_key": "$HF_TOKEN",
                    }
                ],
                "pipeline": {
                    "ingestion": {"run": True},
                },
            }

            with config_path.open("w") as f:
                yaml.dump(legacy_config, f)

            config = load_config(config_path)

            # Verify backward compatibility
            assert len(config.model_list) == 1
            assert config.model_list[0].model_name == "legacy_model"
            assert config.hf_configuration.hf_dataset_name == "legacy_test"

    def test_error_handling_invalid_yaml(self):
        """Test error handling for invalid YAML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.yaml"

            # Create invalid YAML
            config_path.write_text("invalid: yaml: content: [")

            with pytest.raises(ValueError, match="Invalid YAML"):
                load_config(config_path)

    def test_error_handling_missing_file(self):
        """Test error handling for missing configuration files."""
        nonexistent_path = Path("/nonexistent/config.yaml")

        with pytest.raises(FileNotFoundError):
            load_config(nonexistent_path)

    def test_config_with_hf_organization_fallback(self):
        """Test HF_ORGANIZATION fallback to whoami when not set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "hf_fallback_config.yaml"

            config_data = {
                "hf_configuration": {
                    "hf_dataset_name": "fallback_test",
                    "hf_organization": "$HF_ORGANIZATION",  # Will fallback
                    "hf_token": "$HF_TOKEN",
                },
                "model_list": [{"model_name": "test_model"}],
                "pipeline": {"ingestion": {"run": True}},
            }

            with config_path.open("w") as f:
                yaml.dump(config_data, f)

            # Test with HF_TOKEN but no HF_ORGANIZATION
            test_env = {"HF_TOKEN": "hf_test_token"}

            with patch.dict(os.environ, test_env, clear=True):
                with patch("yourbench.utils.configuration_engine.whoami") as mock_whoami:
                    mock_whoami.return_value = {"name": "fallback_user"}

                    config = load_config(config_path)

                    # Should fallback to whoami result
                    assert config.hf_configuration.hf_organization == "fallback_user"

    @patch.dict(os.environ, {"HF_TOKEN": "test_token"})
    def test_path_object_handling(self):
        """Test that Path objects are handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "path_config.yaml"

            config_data = {
                "hf_configuration": {
                    "hf_dataset_name": "path_test",
                    "local_dataset_dir": "data/custom_dataset",
                },
                "model_list": [{"model_name": "test_model"}],
                "pipeline": {
                    "ingestion": {
                        "run": True,
                        "source_documents_dir": "custom/source/path",
                        "output_dir": "custom/output/path",
                    },
                },
            }

            with config_path.open("w") as f:
                yaml.dump(config_data, f)

            config = load_config(config_path)

            # Verify Path objects are created
            assert isinstance(config.hf_configuration.local_dataset_dir, Path)
            assert isinstance(config.pipeline_config.ingestion.source_documents_dir, Path)
            assert isinstance(config.pipeline_config.ingestion.output_dir, Path)

            # Verify values
            assert str(config.hf_configuration.local_dataset_dir) == "data/custom_dataset"
            assert str(config.pipeline_config.ingestion.source_documents_dir) == "custom/source/path"
            assert str(config.pipeline_config.ingestion.output_dir) == "custom/output/path"


class TestConfigurationCLIIntegration:
    """Test CLI integration with the configuration system."""

    @patch("yourbench.config_builder.console")
    def test_cli_create_simple(self, mock_console):
        """Test CLI config creation in simple mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "cli_simple.yaml"

            # This should work without any user input
            config = create_yourbench_config(simple=True)
            save_config(config, config_path)

            # Verify the file was created and is valid
            assert config_path.exists()
            loaded_config = load_config(config_path)
            assert isinstance(loaded_config, YourbenchConfig)

    @patch.dict(os.environ, {"HF_TOKEN": "test_token"})
    def test_config_integration_with_main_module(self):
        """Test that the main module can load and use configurations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "main_integration.yaml"

            # Create a test config
            config = create_yourbench_config(simple=True)
            save_config(config, config_path)

            # Import and test the load functionality
            from yourbench.utils.configuration_engine import YourbenchConfig

            loaded_config = YourbenchConfig.from_yaml(config_path)
            assert isinstance(loaded_config, YourbenchConfig)
            assert len(loaded_config.model_list) > 0
            assert loaded_config.hf_configuration.hf_dataset_name is not None
