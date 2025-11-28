"""
Unified configuration loader for yourbench.

Loads YAML configs and merges with schema defaults using OmegaConf.
"""

import os
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from yourbench.conf.prompts import DEFAULT_PROMPTS, load_prompt
from yourbench.conf.schema import YourbenchConfig, ModelConfig

STAGE_ORDER = [
    "ingestion",
    "summarization",
    "chunking",
    "single_shot_question_generation",
    "multi_hop_question_generation",
    "cross_document_question_generation",
    "question_rewriting",
    "prepare_lighteval",
    "lighteval",
    "citation_score_filtering",
]

# Prompt field paths: (config path tuple, default prompt key)
PROMPT_FIELDS = [
    (("pipeline", "ingestion", "pdf_llm_prompt"), "pdf_llm_prompt"),
    (("pipeline", "summarization", "summarization_user_prompt"), "summarization_user_prompt"),
    (("pipeline", "summarization", "combine_summaries_user_prompt"), "combine_summaries_user_prompt"),
    (("pipeline", "single_shot_question_generation", "single_shot_system_prompt"), "single_shot_system_prompt"),
    (("pipeline", "single_shot_question_generation", "single_shot_system_prompt_multi"), "single_shot_system_prompt_multi"),
    (("pipeline", "single_shot_question_generation", "single_shot_user_prompt"), "single_shot_user_prompt"),
    (("pipeline", "multi_hop_question_generation", "multi_hop_system_prompt"), "multi_hop_system_prompt"),
    (("pipeline", "multi_hop_question_generation", "multi_hop_system_prompt_multi"), "multi_hop_system_prompt_multi"),
    (("pipeline", "multi_hop_question_generation", "multi_hop_user_prompt"), "multi_hop_user_prompt"),
    (("pipeline", "cross_document_question_generation", "multi_hop_system_prompt"), "multi_hop_system_prompt"),
    (("pipeline", "cross_document_question_generation", "multi_hop_system_prompt_multi"), "multi_hop_system_prompt_multi"),
    (("pipeline", "cross_document_question_generation", "multi_hop_user_prompt"), "multi_hop_user_prompt"),
    (("pipeline", "question_rewriting", "question_rewriting_system_prompt"), "question_rewriting_system_prompt"),
    (("pipeline", "question_rewriting", "question_rewriting_user_prompt"), "question_rewriting_user_prompt"),
]


def load_config(yaml_path: str | Path) -> DictConfig:
    """Load a yourbench config from YAML file, merged with schema defaults."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    # Load user config
    user_cfg = OmegaConf.load(yaml_path)

    # Handle legacy field renames before merging
    if "models" in user_cfg and "model_list" not in user_cfg:
        user_cfg["model_list"] = user_cfg.pop("models")
        logger.debug("Renamed 'models' → 'model_list'")

    if "pipeline_config" in user_cfg and "pipeline" not in user_cfg:
        user_cfg["pipeline"] = user_cfg.pop("pipeline_config")

    # Expand $VAR environment variables in user config
    _expand_env_vars(user_cfg)

    # Mark stages as run=True if they're present in user config
    _mark_enabled_stages(user_cfg)

    # Create schema with defaults
    schema = OmegaConf.structured(YourbenchConfig)

    # Merge: schema defaults + user overrides
    cfg = OmegaConf.merge(schema, user_cfg)

    # Set model defaults for each model in list
    _set_model_defaults(cfg)

    # Load prompts from files or package defaults
    _load_prompts(cfg)

    # Assign default model roles
    _assign_model_roles(cfg)

    # Resolve any remaining interpolations
    OmegaConf.resolve(cfg)

    return cfg


def _expand_env_vars(cfg: DictConfig) -> None:
    """Expand $VAR syntax to environment variable values (in-place)."""

    def expand_value(value):
        if not isinstance(value, str):
            return value
        if value.startswith("$") and not value.startswith("${"):
            var_name = value[1:]
            env_value = os.getenv(var_name)
            if env_value is not None:
                return env_value
            if var_name == "HF_ORGANIZATION":
                token = os.getenv("HF_TOKEN")
                if token:
                    try:
                        from huggingface_hub import whoami
                        return whoami(token).get("name", "")
                    except Exception:
                        pass
            logger.debug(f"Environment variable {var_name} not set")
            return ""
        return value

    def walk(node):
        if OmegaConf.is_dict(node):
            for key in list(node.keys()):
                val = node[key]
                if OmegaConf.is_dict(val) or OmegaConf.is_list(val):
                    walk(val)
                else:
                    node[key] = expand_value(val)
        elif OmegaConf.is_list(node):
            for i in range(len(node)):
                val = node[i]
                if OmegaConf.is_dict(val) or OmegaConf.is_list(val):
                    walk(val)
                else:
                    node[i] = expand_value(val)

    walk(cfg)


def _mark_enabled_stages(cfg: DictConfig) -> None:
    """Mark stages as run=True if present in user config (presence = enabled)."""
    if "pipeline" not in cfg:
        return

    for stage in STAGE_ORDER:
        if stage in cfg.pipeline:
            stage_cfg = cfg.pipeline[stage]
            if stage_cfg is None:
                # Empty stage (e.g., "summarization:") means run=True
                cfg.pipeline[stage] = {"run": True}
            elif OmegaConf.is_dict(stage_cfg) and "run" not in stage_cfg:
                stage_cfg["run"] = True


def _set_model_defaults(cfg: DictConfig) -> None:
    """Ensure each model has all required fields with defaults."""
    model_schema = OmegaConf.structured(ModelConfig)

    for i, model in enumerate(cfg.get("model_list", [])):
        # Merge model schema defaults with user-provided values
        cfg.model_list[i] = OmegaConf.merge(model_schema, model)


def _load_prompts(cfg: DictConfig) -> None:
    """Load prompt content from file paths or package defaults."""
    for path_tuple, default_key in PROMPT_FIELDS:
        try:
            node = cfg
            for key in path_tuple[:-1]:
                node = node[key]

            field = path_tuple[-1]
            current_value = node.get(field, "")
            default_path = DEFAULT_PROMPTS.get(default_key, "")

            if current_value:
                # User provided a value - load from path or use as-is
                node[field] = load_prompt(str(current_value), default_path)
            elif default_path:
                # No value - load default prompt
                node[field] = load_prompt("", default_path)
        except Exception:
            pass


def _assign_model_roles(cfg: DictConfig) -> None:
    """Assign default model to stages without explicit model_roles."""
    if not cfg.get("model_list"):
        return

    default_model = cfg.model_list[0].get("model_name", "")
    if not default_model:
        return

    for stage in STAGE_ORDER:
        if stage not in cfg.model_roles:
            cfg.model_roles[stage] = [default_model]


# Helper functions for config access
def get_enabled_stages(cfg: DictConfig) -> list[str]:
    """Return list of enabled pipeline stages in execution order."""
    return [s for s in STAGE_ORDER if cfg.pipeline.get(s, {}).get("run", False)]


def is_stage_enabled(cfg: DictConfig, stage: str) -> bool:
    """Check if a pipeline stage is enabled."""
    return cfg.pipeline.get(stage, {}).get("run", False)


def get_model_for_stage(cfg: DictConfig, stage: str) -> str | None:
    """Get the primary model name for a stage."""
    models = cfg.get("model_roles", {}).get(stage, [])
    if models:
        return models[0]
    if cfg.get("model_list"):
        return cfg.model_list[0].get("model_name")
    return None


def get_model_config(cfg: DictConfig, model_name: str) -> DictConfig | None:
    """Get model config by name."""
    for model in cfg.get("model_list", []):
        if model.get("model_name") == model_name:
            return model
    return None
