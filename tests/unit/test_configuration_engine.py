"""Tests for the modern configuration_engine module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from yourbench.utils.configuration_engine import (
    ModelConfig,
    ChunkingConfig,
    PipelineConfig,
    IngestionConfig,
    YourbenchConfig,
    HuggingFaceConfig,
    SummarizationConfig,
    MultiHopQuestionGenerationConfig,
    SingleShotQuestionGenerationConfig,
    CrossDocumentQuestionGenerationConfig,
    _expand_env,
    _expand_dataclass,
)


class TestEnvironmentVariableExpansion:
    """Test environment variable expansion functionality."""

    def test_expand_env_with_existing_var(self):
        """Test expansion of existing environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = _expand_env("$TEST_VAR")
            assert result == "test_value"

    def test_expand_env_with_nonexistent_var(self):
        """Test expansion of non-existent environment variable."""
        result = _expand_env("$NONEXISTENT_VAR")
        assert result == "$NONEXISTENT_VAR"

    def test_expand_env_non_env_var(self):
        """Test that non-env-var strings are returned unchanged."""
        result = _expand_env("regular_string")
        assert result == "regular_string"

    def test_expand_env_non_string(self):
        """Test that non-string values are returned unchanged."""
        result = _expand_env(123)
        assert result == 123

    def test_expand_env_hf_organization_special_case(self):
        """Test HF_ORGANIZATION special case fallback."""
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}):
            with patch("yourbench.utils.configuration_engine.whoami") as mock_whoami:
                mock_whoami.return_value = {"name": "test_user"}
                result = _expand_env("$HF_ORGANIZATION")
                assert result == "test_user"

    def test_expand_env_hf_organization_fallback_failure(self):
        """Test HF_ORGANIZATION fallback when whoami fails."""
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}):
            with patch("yourbench.utils.configuration_engine.whoami") as mock_whoami:
                mock_whoami.side_effect = Exception("API error")
                result = _expand_env("$HF_ORGANIZATION")
                assert result == "$HF_ORGANIZATION"

    def test_expand_dataclass(self):
        """Test dataclass field expansion."""
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            field1: str = "$TEST_VAR"
            field2: int = 42

        with patch.dict(os.environ, {"TEST_VAR": "expanded_value"}):
            obj = TestClass()
            _expand_dataclass(obj)
            assert obj.field1 == "expanded_value"
            assert obj.field2 == 42


class TestHuggingFaceConfig:
    """Test HuggingFaceConfig dataclass."""

    def test_default_initialization(self):
        """Test HuggingFaceConfig with default values."""
        config = HuggingFaceConfig()
        assert not config.private
        assert not config.concat_if_exist
        assert config.local_saving
        assert config.upload_card

    def test_env_variable_expansion(self):
        """Test that environment variables are expanded during initialization."""
        with patch.dict(os.environ, {"HF_TOKEN": "test_token", "HF_ORGANIZATION": "test_org"}):
            config = HuggingFaceConfig()
            assert config.hf_token == "test_token"
            assert config.hf_organization == "test_org"

    def test_custom_values(self):
        """Test HuggingFaceConfig with custom values."""
        config = HuggingFaceConfig(
            hf_dataset_name="custom_dataset",
            private=True,
            concat_if_exist=True,
            local_saving=False,
        )
        assert config.hf_dataset_name == "custom_dataset"
        assert config.private
        assert config.concat_if_exist
        assert not config.local_saving


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_initialization(self):
        """Test ModelConfig with default values."""
        config = ModelConfig()
        assert config.max_concurrent_requests == 32
        assert config.encoding_name == "cl100k_base"
        assert config.provider == "auto"  # Auto-assigned when no base_url

    def test_with_base_url(self):
        """Test ModelConfig with base_url."""
        config = ModelConfig(base_url="https://api.example.com")
        assert config.base_url == "https://api.example.com"
        assert config.provider is None  # No auto-assignment with base_url

    def test_with_provider(self):
        """Test ModelConfig with explicit provider."""
        config = ModelConfig(provider="fireworks-ai")
        assert config.provider == "fireworks-ai"

    def test_env_variable_expansion(self):
        """Test that environment variables are expanded."""
        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            config = ModelConfig()
            assert config.api_key == "test_token"


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_initialization(self):
        """Test PipelineConfig with default values."""
        config = PipelineConfig()
        assert isinstance(config.ingestion, IngestionConfig)
        assert isinstance(config.summarization, SummarizationConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.single_shot_question_generation, SingleShotQuestionGenerationConfig)
        assert isinstance(config.multi_hop_question_generation, MultiHopQuestionGenerationConfig)
        assert isinstance(config.cross_document_question_generation, CrossDocumentQuestionGenerationConfig)

    def test_nested_config_initialization(self):
        """Test that nested configs are properly initialized."""
        config = PipelineConfig()
        assert not config.ingestion.run
        assert not config.summarization.run
        assert not config.chunking.run


class TestIngestionConfig:
    """Test IngestionConfig dataclass."""

    def test_default_initialization(self):
        """Test IngestionConfig with default values."""
        config = IngestionConfig()
        assert not config.run
        assert config.upload_to_hub
        assert not config.llm_ingestion
        assert config.pdf_dpi == 300
        assert isinstance(config.source_documents_dir, Path)
        assert isinstance(config.output_dir, Path)

    def test_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = IngestionConfig(source_documents_dir="custom/source", output_dir="custom/output")
        assert isinstance(config.source_documents_dir, Path)
        assert isinstance(config.output_dir, Path)
        assert str(config.source_documents_dir) == "custom/source"
        assert str(config.output_dir) == "custom/output"

    def test_prompt_loading(self):
        """Test that prompt files are loaded if they exist."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Test prompt content")
            f.flush()

            config = IngestionConfig(pdf_llm_prompt=f.name)
            assert config.pdf_llm_prompt == "Test prompt content"

            # Clean up
            os.unlink(f.name)


class TestChunkingConfig:
    """Test ChunkingConfig dataclass."""

    def test_default_initialization(self):
        """Test ChunkingConfig with default values."""
        config = ChunkingConfig()
        assert not config.run
        assert config.l_max_tokens == 8192
        assert config.token_overlap == 512
        assert config.encoding_name == "cl100k_base"
        assert config.h_min == 2
        assert config.h_max == 5
        assert config.num_multihops_factor == 1

    def test_custom_values(self):
        """Test ChunkingConfig with custom values."""
        config = ChunkingConfig(l_max_tokens=1024, h_min=3, h_max=7, num_multihops_factor=2)
        assert config.l_max_tokens == 1024
        assert config.h_min == 3
        assert config.h_max == 7
        assert config.num_multihops_factor == 2


class TestQuestionGenerationConfigs:
    """Test question generation configuration classes."""

    def test_single_shot_config(self):
        """Test SingleShotQuestionGenerationConfig."""
        config = SingleShotQuestionGenerationConfig()
        assert not config.run
        assert config.question_mode == "open-ended"

    def test_multi_hop_config(self):
        """Test MultiHopQuestionGenerationConfig."""
        config = MultiHopQuestionGenerationConfig()
        assert not config.run
        assert config.question_mode == "open-ended"

    def test_cross_document_config(self):
        """Test CrossDocumentQuestionGenerationConfig."""
        config = CrossDocumentQuestionGenerationConfig()
        assert not config.run
        assert config.max_combinations == 100
        assert config.chunks_per_document == 1
        assert config.num_docs_per_combination == [2, 5]
        assert config.random_seed == 42


class TestYourbenchConfig:
    """Test the main YourbenchConfig class."""

    def test_default_initialization(self):
        """Test YourbenchConfig with default values."""
        config = YourbenchConfig()
        assert isinstance(config.hf_configuration, HuggingFaceConfig)
        assert isinstance(config.pipeline_config, PipelineConfig)
        assert config.model_list == []
        assert config.model_roles == {}
        assert not config.debug

    def test_model_role_assignment(self):
        """Test automatic model role assignment."""
        model = ModelConfig(model_name="test_model")
        config = YourbenchConfig(model_list=[model])

        # Check that model roles are assigned
        assert "ingestion" in config.model_roles
        assert "summarization" in config.model_roles
        assert config.model_roles["ingestion"] == ["test_model"]
        assert config.model_roles["summarization"] == ["test_model"]

    def test_no_model_role_assignment_without_models(self):
        """Test that no model roles are assigned when no models are provided."""
        config = YourbenchConfig(model_list=[])
        # Should have default empty model_roles
        assert len(config.model_roles) == 0

    def test_from_yaml_basic(self):
        """Test loading configuration from YAML."""
        yaml_content = {
            "hf_configuration": {
                "hf_dataset_name": "test_dataset",
                "private": True,
            },
            "model_list": [
                {
                    "model_name": "test_model",
                    "api_key": "$HF_TOKEN",
                    "max_concurrent_requests": 16,
                }
            ],
            "model_roles": {
                "ingestion": ["test_model"],
                "summarization": ["test_model"],
            },
            "pipeline": {
                "ingestion": {
                    "run": True,
                    "source_documents_dir": "test/source",
                },
                "summarization": {
                    "run": True,
                },
                "chunking": {
                    "run": True,
                    "l_max_tokens": 1024,
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()

            config = YourbenchConfig.from_yaml(f.name)

            # Test HF configuration
            assert config.hf_configuration.hf_dataset_name == "test_dataset"
            assert config.hf_configuration.private

            # Test model configuration
            assert len(config.model_list) == 1
            assert config.model_list[0].model_name == "test_model"
            assert config.model_list[0].max_concurrent_requests == 16

            # Test model roles
            assert config.model_roles["ingestion"] == ["test_model"]
            assert config.model_roles["summarization"] == ["test_model"]

            # Test pipeline configuration
            assert config.pipeline_config.ingestion.run
            assert str(config.pipeline_config.ingestion.source_documents_dir) == "test/source"
            assert config.pipeline_config.summarization.run
            assert config.pipeline_config.chunking.run
            assert config.pipeline_config.chunking.l_max_tokens == 1024

            # Clean up
            os.unlink(f.name)

    def test_from_yaml_with_defaults(self):
        """Test that missing pipeline stages get default configurations."""
        yaml_content = {
            "hf_configuration": {
                "hf_dataset_name": "test_dataset",
            },
            "model_list": [
                {
                    "model_name": "test_model",
                }
            ],
            "pipeline": {
                "ingestion": {
                    "run": True,
                },
                # Missing other stages should get defaults
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()

            config = YourbenchConfig.from_yaml(f.name)

            # Test that missing stages get default configs
            assert config.pipeline_config.ingestion.run
            assert not config.pipeline_config.summarization.run  # Default
            assert not config.pipeline_config.chunking.run  # Default

            # Clean up
            os.unlink(f.name)

    def test_from_yaml_stage_run_defaults(self):
        """Test that stages present in config default to run=True."""
        yaml_content = {
            "hf_configuration": {
                "hf_dataset_name": "test_dataset",
            },
            "model_list": [
                {
                    "model_name": "test_model",
                }
            ],
            "pipeline": {
                "ingestion": {},  # No explicit run: false
                "summarization": {},  # No explicit run: false
                "chunking": {"run": False},  # Explicit run: false
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()

            config = YourbenchConfig.from_yaml(f.name)

            # Test that stages present in config default to run=True
            assert config.pipeline_config.ingestion.run  # Default for present stage
            assert config.pipeline_config.summarization.run  # Default for present stage
            assert not config.pipeline_config.chunking.run  # Explicit false

            # Clean up
            os.unlink(f.name)

    def test_from_yaml_backward_compatibility(self):
        """Test backward compatibility with 'models' key."""
        yaml_content = {
            "hf_configuration": {
                "hf_dataset_name": "test_dataset",
            },
            "models": [  # Old key name
                {
                    "model_name": "test_model",
                }
            ],
            "pipeline": {
                "ingestion": {"run": True},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()

            config = YourbenchConfig.from_yaml(f.name)

            # Test that old 'models' key is handled
            assert len(config.model_list) == 1
            assert config.model_list[0].model_name == "test_model"

            # Clean up
            os.unlink(f.name)


class TestConfigurationIntegration:
    """Integration tests for the configuration system."""

    def test_full_configuration_cycle(self):
        """Test creating, saving, and loading a complete configuration."""
        # Create a full configuration
        hf_config = HuggingFaceConfig(
            hf_dataset_name="integration_test",
            private=False,
        )

        models = [
            ModelConfig(
                model_name="test_model_1",
                provider="fireworks-ai",
                max_concurrent_requests=16,
            ),
            ModelConfig(
                model_name="test_model_2",
                base_url="https://api.example.com",
                api_key="$API_KEY",
                max_concurrent_requests=8,
            ),
        ]

        model_roles = {
            "ingestion": ["test_model_1"],
            "summarization": ["test_model_2"],
            "single_shot_question_generation": ["test_model_1"],
            "multi_hop_question_generation": ["test_model_2"],
        }

        pipeline_config = PipelineConfig(
            ingestion=IngestionConfig(run=True),
            summarization=SummarizationConfig(run=True),
            chunking=ChunkingConfig(run=True, l_max_tokens=1024),
            single_shot_question_generation=SingleShotQuestionGenerationConfig(run=True),
            multi_hop_question_generation=MultiHopQuestionGenerationConfig(run=True),
        )

        original_config = YourbenchConfig(
            hf_configuration=hf_config,
            model_list=models,
            model_roles=model_roles,
            pipeline_config=pipeline_config,
        )

        # Convert to dict and save as YAML
        config_dict = {
            "hf_configuration": {
                "hf_dataset_name": original_config.hf_configuration.hf_dataset_name,
                "private": original_config.hf_configuration.private,
            },
            "model_list": [
                {
                    "model_name": model.model_name,
                    "provider": model.provider,
                    "base_url": model.base_url,
                    "api_key": model.api_key,
                    "max_concurrent_requests": model.max_concurrent_requests,
                }
                for model in original_config.model_list
            ],
            "model_roles": original_config.model_roles,
            "pipeline": {
                "ingestion": {"run": original_config.pipeline_config.ingestion.run},
                "summarization": {"run": original_config.pipeline_config.summarization.run},
                "chunking": {
                    "run": original_config.pipeline_config.chunking.run,
                    "l_max_tokens": original_config.pipeline_config.chunking.l_max_tokens,
                },
                "single_shot_question_generation": {
                    "run": original_config.pipeline_config.single_shot_question_generation.run
                },
                "multi_hop_question_generation": {
                    "run": original_config.pipeline_config.multi_hop_question_generation.run
                },
            },
        }

        # Clean up None values
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items() if v is not None}
            elif isinstance(d, list):
                return [clean_dict(item) for item in d]
            return d

        config_dict = clean_dict(config_dict)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            f.flush()

            # Load the configuration back
            loaded_config = YourbenchConfig.from_yaml(f.name)

            # Verify the loaded configuration matches original
            assert loaded_config.hf_configuration.hf_dataset_name == "integration_test"
            assert not loaded_config.hf_configuration.private

            assert len(loaded_config.model_list) == 2
            assert loaded_config.model_list[0].model_name == "test_model_1"
            assert loaded_config.model_list[1].model_name == "test_model_2"

            assert loaded_config.model_roles["ingestion"] == ["test_model_1"]
            assert loaded_config.model_roles["summarization"] == ["test_model_2"]

            assert loaded_config.pipeline_config.ingestion.run
            assert loaded_config.pipeline_config.summarization.run
            assert loaded_config.pipeline_config.chunking.run
            assert loaded_config.pipeline_config.chunking.l_max_tokens == 1024

            # Clean up
            os.unlink(f.name)

    def test_configuration_validation(self):
        """Test that configuration validation works correctly."""
        # Test that configuration accepts valid values
        config = ModelConfig(model_name="test_model", max_concurrent_requests=32)
        assert config.model_name == "test_model"
        assert config.max_concurrent_requests == 32

        # Test that invalid types are still created (dataclass doesn't validate types by default)
        # This is expected behavior - type hints are for static analysis
        config2 = ModelConfig(model_name=None, max_concurrent_requests="invalid")
        assert config2.model_name is None
        assert config2.max_concurrent_requests == "invalid"