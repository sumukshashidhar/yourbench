"""Unit tests for configuration loading."""

from omegaconf import OmegaConf

from yourbench.conf.loader import load_config, get_enabled_stages


def test_load_basic_config(tmp_path):
    """Basic config loads correctly."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
hf_configuration:
  hf_dataset_name: test-dataset
model_list:
  - model_name: test-model
pipeline:
  ingestion:
    run: true
""",
        encoding="utf-8",
    )

    cfg = load_config(yaml_path)

    assert cfg.hf_configuration.hf_dataset_name == "test-dataset"
    assert cfg.pipeline.ingestion.run is True
    assert get_enabled_stages(cfg) == ["ingestion"]


def test_stage_presence_implies_run(tmp_path):
    """Stage presence (even without run: true) means it's enabled."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
hf_configuration:
  hf_dataset_name: test-dataset
model_list:
  - model_name: test-model
pipeline:
  ingestion:
    source_documents_dir: data/raw
  summarization:
  chunking:
""",
        encoding="utf-8",
    )

    cfg = load_config(yaml_path)
    enabled = get_enabled_stages(cfg)

    assert "ingestion" in enabled
    assert "summarization" in enabled
    assert "chunking" in enabled
    # Not present means disabled
    assert "single_shot_question_generation" not in enabled


def test_legacy_field_renames(tmp_path):
    """Legacy 'models' and 'pipeline_config' are renamed."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
hf_configuration:
  hf_dataset_name: test-dataset
models:
  - model_name: old-model-name
pipeline_config:
  ingestion:
    run: true
""",
        encoding="utf-8",
    )

    cfg = load_config(yaml_path)

    # Should have been renamed
    assert cfg.model_list[0].model_name == "old-model-name"
    assert cfg.pipeline.ingestion.run is True


def test_config_to_dict_roundtrip(tmp_path):
    """Config can be converted to dict."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
hf_configuration:
  hf_dataset_name: my-dataset
""",
        encoding="utf-8",
    )

    cfg = load_config(yaml_path)
    data = OmegaConf.to_container(cfg, resolve=True)
    assert data["hf_configuration"]["hf_dataset_name"] == "my-dataset"


def test_model_roles_default_assignment(tmp_path):
    """Model roles are auto-assigned if not specified."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
hf_configuration:
  hf_dataset_name: test-dataset
model_list:
  - model_name: my-model
pipeline:
  ingestion:
""",
        encoding="utf-8",
    )

    cfg = load_config(yaml_path)

    # model_roles should be auto-assigned
    assert cfg.model_roles.ingestion == ["my-model"]
