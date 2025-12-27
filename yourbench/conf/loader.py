"""
Simplified configuration loader using Pydantic for validation.

Loads YAML configs and validates against Pydantic schemas.
"""

import os
from typing import Any
from pathlib import Path

import yaml
from loguru import logger

from yourbench.conf.schema import (
    ModelConfig,
    YourbenchConfig,
    ConfigValidationError,
    _expand_env,
)
from yourbench.conf.prompts import DEFAULT_PROMPTS, load_prompt


STAGE_ORDER = [
    "ingestion",
    "summarization",
    "chunking",
    "single_shot_question_generation",
    "multi_hop_question_generation",
    "cross_document_question_generation",
    "question_rewriting",
    "prepare_lighteval",
    "citation_score_filtering",
]

# Prompt field paths: (config path tuple, default prompt key)
PROMPT_FIELDS = [
    (("pipeline", "ingestion", "pdf_llm_prompt"), "pdf_llm_prompt"),
    (("pipeline", "summarization", "summarization_user_prompt"), "summarization_user_prompt"),
    (("pipeline", "summarization", "combine_summaries_user_prompt"), "combine_summaries_user_prompt"),
    (("pipeline", "single_shot_question_generation", "single_shot_system_prompt"), "single_shot_system_prompt"),
    (
        ("pipeline", "single_shot_question_generation", "single_shot_system_prompt_multi"),
        "single_shot_system_prompt_multi",
    ),
    (("pipeline", "single_shot_question_generation", "single_shot_user_prompt"), "single_shot_user_prompt"),
    (("pipeline", "multi_hop_question_generation", "multi_hop_system_prompt"), "multi_hop_system_prompt"),
    (("pipeline", "multi_hop_question_generation", "multi_hop_system_prompt_multi"), "multi_hop_system_prompt_multi"),
    (("pipeline", "multi_hop_question_generation", "multi_hop_user_prompt"), "multi_hop_user_prompt"),
    (("pipeline", "cross_document_question_generation", "multi_hop_system_prompt"), "multi_hop_system_prompt"),
    (
        ("pipeline", "cross_document_question_generation", "multi_hop_system_prompt_multi"),
        "multi_hop_system_prompt_multi",
    ),
    (("pipeline", "cross_document_question_generation", "multi_hop_user_prompt"), "multi_hop_user_prompt"),
    (("pipeline", "question_rewriting", "question_rewriting_system_prompt"), "question_rewriting_system_prompt"),
    (("pipeline", "question_rewriting", "question_rewriting_user_prompt"), "question_rewriting_user_prompt"),
]


def load_config(yaml_path: str | Path) -> YourbenchConfig:
    """Load a yourbench config from YAML file.

    1. Parse YAML
    2. Expand $VAR environment variables
    3. Handle legacy field names
    4. Mark enabled stages (presence = run)
    5. Auto-load OpenAI from env if no models
    6. Validate with Pydantic
    7. Load prompts
    8. Assign model roles
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    # Load raw YAML
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    # Transform the data
    data = _handle_legacy_fields(data)
    data = _expand_env_vars(data)
    data = _mark_enabled_stages(data)
    data = _auto_load_openai_from_env(data)

    # Validate with Pydantic
    try:
        config = YourbenchConfig.model_validate(data)
    except Exception as e:
        raise ConfigValidationError(f"Config validation failed: {e}") from e

    # Post-processing (needs access to the validated config)
    _load_prompts(config)
    _assign_model_roles(config)

    return config


def _handle_legacy_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Handle legacy field renames."""
    if "models" in data and "model_list" not in data:
        data["model_list"] = data.pop("models")
        logger.debug("Renamed 'models' -> 'model_list'")

    if "pipeline_config" in data and "pipeline" not in data:
        data["pipeline"] = data.pop("pipeline_config")

    return data


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand $VAR syntax in data."""
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    elif isinstance(data, str):
        return _expand_env(data)
    return data


def _mark_enabled_stages(data: dict[str, Any]) -> dict[str, Any]:
    """Mark stages as run=True if present in config (presence = enabled)."""
    pipeline = data.get("pipeline", {})
    if not pipeline:
        return data

    for stage in STAGE_ORDER:
        if stage in pipeline:
            stage_cfg = pipeline[stage]
            if stage_cfg is None:
                # Empty stage (e.g., "summarization:") means run=True
                pipeline[stage] = {"run": True}
            elif isinstance(stage_cfg, dict) and "run" not in stage_cfg:
                stage_cfg["run"] = True

    data["pipeline"] = pipeline
    return data


def _auto_load_openai_from_env(data: dict[str, Any]) -> dict[str, Any]:
    """Create OpenAI model from env vars if no models configured."""
    if data.get("model_list") or data.get("models"):
        return data

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return data

    openai_model = {
        "model_name": os.getenv("OPENAI_MODEL", "gpt-4"),
        "api_key": "$OPENAI_API_KEY",
        "max_concurrent_requests": 8,
    }

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        openai_model["base_url"] = base_url

    data["model_list"] = [openai_model]
    logger.info(f"Auto-loaded OpenAI model from environment: {openai_model['model_name']}")
    return data


def _load_prompts(config: YourbenchConfig) -> None:
    """Load prompt content from file paths or package defaults."""
    for path_tuple, default_key in PROMPT_FIELDS:
        try:
            # Navigate to the parent object
            obj = config
            for key in path_tuple[:-1]:
                obj = getattr(obj, key)

            field = path_tuple[-1]
            current_value = getattr(obj, field, "")
            default_path = DEFAULT_PROMPTS.get(default_key, "")

            if current_value:
                # User provided a value - load from path or use as-is
                new_value = load_prompt(str(current_value), default_path)
            elif default_path:
                # No value - load default prompt
                new_value = load_prompt("", default_path)
            else:
                continue

            # Set the value on the Pydantic model
            setattr(obj, field, new_value)
        except Exception as exc:
            logger.warning("Failed to load prompt", path=".".join(path_tuple), default=default_key, error=exc)


def _assign_model_roles(config: YourbenchConfig) -> None:
    """Assign default model to stages without explicit model_roles."""
    if not config.model_list:
        return

    default_model = config.model_list[0].model_name
    if not default_model:
        return

    for stage in STAGE_ORDER:
        if stage not in config.model_roles:
            config.model_roles[stage] = [default_model]


# Helper functions for config access
def get_enabled_stages(config: YourbenchConfig) -> list[str]:
    """Return list of enabled pipeline stages in execution order."""
    return [
        s
        for s in STAGE_ORDER
        if getattr(config.pipeline, s, None) and getattr(getattr(config.pipeline, s), "run", False)
    ]


def is_stage_enabled(config: YourbenchConfig, stage: str) -> bool:
    """Check if a pipeline stage is enabled."""
    stage_cfg = getattr(config.pipeline, stage, None)
    return stage_cfg and getattr(stage_cfg, "run", False)


def get_model_for_stage(config: YourbenchConfig, stage: str) -> str | None:
    """Get the primary model name for a stage."""
    models = config.model_roles.get(stage, [])
    if models:
        return models[0]
    if config.model_list:
        return config.model_list[0].model_name
    return None


def get_model_config(config: YourbenchConfig, model_name: str) -> ModelConfig | None:
    """Get model config by name."""
    for model in config.model_list:
        if model.model_name == model_name:
            return model
    return None
