"""Dataset card generation and upload functionality for HuggingFace Hub."""

import os
from copy import deepcopy
from typing import Any
from pathlib import Path

from loguru import logger

from huggingface_hub import DatasetCard, DatasetCardData
from yourbench.conf.prompts import load_prompt_from_package as _load_prompt_from_package


def _is_offline() -> bool:
    """Check if offline mode enabled."""
    return os.environ.get("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes")


# Dataset card generation functions


def extract_readme_metadata(repo_id: str, token: str | None = None) -> str:
    """Extracts the metadata from the README.md file of the dataset repository.
    We have to download the previous README.md file in the repo, extract the metadata from it.
    Args:
        repo_id: The ID of the repository to push to, from the `push_to_hub` method.
        token: The token to authenticate with the Hugging Face Hub, from the `push_to_hub` method.
    Returns:
        The metadata extracted from the README.md file of the dataset repository as a str.
    """
    try:
        import re
        from pathlib import Path

        from huggingface_hub.file_download import hf_hub_download

        readme_path = Path(hf_hub_download(repo_id, "README.md", repo_type="dataset", token=token))
        # Extract the content between the '---' markers
        metadata_match = re.findall(r"---\n(.*?)\n---", readme_path.read_text(), re.DOTALL)

        if not metadata_match:
            logger.debug("No YAML metadata found in the README.md")
            return ""

        return metadata_match[0]

    except Exception as e:
        logger.debug(f"Failed to extract metadata from README.md: {e}")
        return ""


def extract_dataset_info(repo_id: str, token: str | None = None) -> str:
    """
    Extract dataset_info section from README metadata.

    Args:
        repo_id: The dataset repository ID
        token: Optional HuggingFace token for authentication

    Returns:
        The dataset_info section as a string, or empty string if not found
    """
    readme_metadata = extract_readme_metadata(repo_id=repo_id, token=token)
    if not readme_metadata:
        return ""

    section_prefix = "dataset_info:"
    if section_prefix not in readme_metadata:
        return ""

    try:
        # Extract the part after `dataset_info:` prefix
        config_data = section_prefix + readme_metadata.split(section_prefix)[1]
        return config_data
    except IndexError:
        logger.debug("Failed to extract dataset_info section from metadata")
        return ""


def _serialize_config_for_card(config: Any) -> str:
    """
    Sanitize and serialize pipeline config to YAML for inclusion in dataset card.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config serialization")

    # Load default prompts to compare against

    # Map of prompt fields to their default package paths
    default_prompt_paths = {
        "pdf_llm_prompt": "ingestion/pdf_llm_prompt.md",
        "summarization_user_prompt": "summarization/summarization_user_prompt.md",
        "combine_summaries_user_prompt": "summarization/combine_summaries_user_prompt.md",
        "single_shot_system_prompt": "question_generation/single_shot_system_prompt.md",
        "single_shot_system_prompt_multi": "question_generation/single_shot_system_prompt_multi.md",
        "single_shot_user_prompt": "question_generation/single_shot_user_prompt.md",
        "multi_hop_system_prompt": "question_generation/multi_hop_system_prompt.md",
        "multi_hop_user_prompt": "question_generation/multi_hop_user_prompt.md",
        "question_rewriting_system_prompt": "question_rewriting/question_rewriting_system_prompt.md",
        "question_rewriting_user_prompt": "question_rewriting/question_rewriting_user_prompt.md",
    }

    # Load default prompts for comparison
    default_prompts = {}
    for field, path in default_prompt_paths.items():
        content = _load_prompt_from_package(path)
        if content:
            default_prompts[field] = content

    def _is_default_prompt(value: str, field_name: str) -> bool:
        """Check if a prompt value matches the default."""
        if field_name in default_prompts:
            return value.strip() == default_prompts[field_name].strip()
        return False

    def _make_relative_path(path_str: str) -> str:
        """Convert absolute path to relative if possible."""
        try:
            path = Path(path_str)
            # If it's already relative, return as is
            if not path.is_absolute():
                return path_str

            # For absolute paths, try to make relative to cwd
            cwd = Path.cwd()

            # Handle paths that might not exist yet
            if path.exists():
                abs_path = path.resolve()
                try:
                    rel_path = abs_path.relative_to(cwd)
                    return str(rel_path)
                except ValueError:
                    pass
            else:
                # For non-existent paths, do string-based relative conversion
                cwd_str = str(cwd)
                if path_str.startswith(cwd_str):
                    return path_str[len(cwd_str) :].lstrip("/\\")

            # If we can't make it relative, return just the last parts
            # This helps avoid exposing full system paths
            parts = path.parts
            if len(parts) > 3:
                # Keep last 3 parts for context
                return str(Path(*parts[-3:]))

            return path_str
        except Exception:
            # If all else fails, return as is
            return path_str

    def _sanitize(obj, key=None, parent_key=None):
        if isinstance(obj, dict):
            sanitized_dict = {}
            for k, v in obj.items():
                sanitized_value = _sanitize(v, k, key)
                # Skip fields with None values or empty strings
                if sanitized_value is not None and sanitized_value != "":
                    sanitized_dict[k] = sanitized_value
            return sanitized_dict if sanitized_dict else None

        if isinstance(obj, list):
            return [_sanitize(v, key, parent_key) for v in obj]

        if isinstance(obj, Path):
            # Convert Path objects to relative strings
            return _make_relative_path(str(obj))

        if isinstance(obj, str):
            # Keep placeholders
            if obj.startswith("$"):
                return obj

            # Handle paths - make them relative
            if key and any(path_key in key.lower() for path_key in ["path", "dir", "directory"]):
                if "/" in obj or "\\" in obj:
                    return _make_relative_path(obj)

            # Handle prompt fields
            if key and "prompt" in key.lower():
                # Check if it's a default prompt
                if _is_default_prompt(obj, key):
                    # Return None to filter out default prompts entirely
                    return None

                # All non-default prompts are custom
                return f"custom_{key}.md"

            # Mask api_key arguments
            if key and "api_key" in key.lower():
                return "$API_KEY"
            # Mask OpenAI API keys
            if obj.startswith("sk-"):
                return "$OPENAI_API_KEY"
            # Mask HuggingFace tokens
            if obj.startswith("hf_"):
                return "$HF_TOKEN"
            # Mask HF organization/username in hf_organization field
            if key == "hf_organization" and not obj.startswith("$"):
                return "$HF_ORGANISATION"
            return obj

        # Explicitly return boolean, integer, float values unchanged
        if isinstance(obj, (bool, int, float)):
            return obj

        # Return None for None values (will be filtered out)
        if obj is None:
            return None

        return obj

    # Convert config to dict for serialization
    from omegaconf import OmegaConf, DictConfig

    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    elif hasattr(config, "model_dump"):
        config_dict = config.model_dump()
    elif hasattr(config, "__dataclass_fields__"):
        from dataclasses import asdict

        config_dict = {k: v for k, v in asdict(config).items() if not k.startswith("_")}
    else:
        config_dict = dict(config) if hasattr(config, "items") else config

    # First pass sanitization
    sanitized = _sanitize(deepcopy(config_dict))

    # Remove empty dictionaries and None values recursively
    def _remove_empty(obj):
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                cleaned_value = _remove_empty(v)
                if cleaned_value is not None and cleaned_value != {} and cleaned_value != []:
                    cleaned[k] = cleaned_value
            return cleaned if cleaned else None
        elif isinstance(obj, list):
            cleaned = [_remove_empty(item) for item in obj]
            return [item for item in cleaned if item is not None]
        else:
            return obj

    sanitized = _remove_empty(sanitized)

    # Filter out default values from hf_configuration
    if "hf_configuration" in sanitized:
        hf_config = sanitized["hf_configuration"]
        # Remove default values
        defaults_to_remove = {
            "private": False,
            "concat_if_exist": False,
            "local_saving": True,
            "upload_card": True,
            "export_jsonl": False,
            "local_dataset_dir": "data/saved_dataset",
            "jsonl_export_dir": "data/jsonl_export",
        }
        for key, default_value in defaults_to_remove.items():
            if key in hf_config and hf_config[key] == default_value:
                del hf_config[key]

    # Filter out default values from model_list
    if "model_list" in sanitized:
        model_list = sanitized["model_list"]
        if isinstance(model_list, list):
            for model in model_list:
                if isinstance(model, dict):
                    # Remove model-level defaults
                    model_defaults = {
                        "max_concurrent_requests": 32,
                        "encoding_name": "cl100k_base",
                    }
                    for key, default_value in model_defaults.items():
                        if key in model and model[key] == default_value:
                            del model[key]

    # Filter out default values from pipeline stages
    # Handle both 'pipeline' and 'pipeline_config' keys for backward compatibility
    pipeline_key = "pipeline_config" if "pipeline_config" in sanitized else "pipeline"
    if pipeline_key in sanitized:
        pipeline = sanitized[pipeline_key]
        # Remove stages that are not enabled
        stages_to_remove = []
        for stage, stage_config in pipeline.items():
            if isinstance(stage_config, dict):
                # Remove run: false stages entirely
                if stage_config.get("run") is False:
                    stages_to_remove.append(stage)
                # Remove run: true as it's redundant when stage is present
                elif stage_config.get("run") is True:
                    del stage_config["run"]

                # Remove other stage-specific defaults
                stage_defaults = {
                    # Ingestion defaults
                    "upload_to_hub": True,
                    "llm_ingestion": False,
                    "pdf_dpi": 300,
                    # Summarization defaults
                    "max_tokens": 32768,
                    "token_overlap": 512,
                    "encoding_name": "cl100k_base",
                    # Chunking defaults
                    "l_max_tokens": 8192,
                    "h_min": 2,
                    "h_max": 5,
                    "num_multihops_factor": 1,
                    # Cross-document defaults
                    "max_combinations": 100,
                    "chunks_per_document": 1,
                    "num_docs_per_combination": [2, 5],
                    "random_seed": 42,
                    # Citation filtering defaults
                    "subset": "prepared_lighteval",
                    "alpha": 0.7,
                    "beta": 0.3,
                    # Question generation defaults
                    "question_mode": "open-ended",
                    # Default file extensions
                    "supported_file_extensions": [
                        ".md",
                        ".txt",
                        ".html",
                        ".htm",
                        ".pdf",
                        ".docx",
                        ".doc",
                        ".pptx",
                        ".ppt",
                        ".xlsx",
                        ".xls",
                        ".rtf",
                        ".odt",
                    ],
                }

                for key, default_value in stage_defaults.items():
                    if key in stage_config and stage_config[key] == default_value:
                        del stage_config[key]

        for stage in stages_to_remove:
            del pipeline[stage]

    # Handle model_roles - if all roles use the same single model, remove it
    if "model_roles" in sanitized:
        model_roles = sanitized["model_roles"]
        # Get all unique models across all roles
        all_models = set()
        for role_models in model_roles.values():
            if isinstance(role_models, list):
                all_models.update(role_models)

        # If there's only one model used everywhere, remove model_roles entirely
        if len(all_models) <= 1:
            del sanitized["model_roles"]

    # Remove debug: false as it's the default
    if sanitized.get("debug") is False:
        del sanitized["debug"]

    # Rename pipeline_config to pipeline for YAML compatibility
    if "pipeline_config" in sanitized:
        sanitized["pipeline"] = sanitized.pop("pipeline_config")

    # Reorder sections: hf_configuration, model_list, model_roles, pipeline, then everything else
    ordered_config = {}
    if "hf_configuration" in sanitized:
        ordered_config["hf_configuration"] = sanitized.pop("hf_configuration")
    if "model_list" in sanitized:
        ordered_config["model_list"] = sanitized.pop("model_list")
    if "model_roles" in sanitized:
        ordered_config["model_roles"] = sanitized.pop("model_roles")
    if "pipeline" in sanitized:
        ordered_config["pipeline"] = sanitized.pop("pipeline")
    # Add remaining sections
    ordered_config.update(sanitized)

    return yaml.safe_dump(ordered_config, sort_keys=False, default_flow_style=False)


def _get_pipeline_subset_info(config: Any) -> str:
    """
    Generate a formatted markdown list of enabled pipeline stages with descriptions.
    The resulting markdown is used in the dataset card to document
    which processing steps were included in the pipeline.

    Args:
        config: The complete pipeline configuration dictionary containing
               the 'pipeline' section with enabled stages

    Returns:
        str: A markdown-formatted string with bullet points for each enabled pipeline stage,
             or an empty string if no stages are enabled
    """

    mapping = {
        "ingestion": "Read raw source documents, convert them to normalized markdown and save for downstream steps",
        "upload_ingest_to_hub": "Package and push ingested markdown dataset to the Hugging Face Hub or save locally with standardized fields",
        "summarization": "Perform hierarchical summarization: chunk-level LLM summaries followed by combine-stage reduction",
        "chunking": "Split texts into token-based single-hop and multi-hop chunks",
        "single_shot_question_generation": "Generate standalone question-answer pairs per chunk using LLM",
        "multi_hop_question_generation": "Generate multi-hop QA pairs requiring reasoning across multiple chunks",
        "lighteval": "Merge QA pairs and chunk metadata into a lighteval compatible dataset for quick model-based scoring",
        "citation_score_filtering": "Compute overlap-based citation scores and filter QA pairs accordingly",
    }
    # Get pipeline config - supports both attribute and dict access
    pipeline = config.pipeline
    lines = []
    for stage_name in [
        "ingestion",
        "summarization",
        "chunking",
        "single_shot_question_generation",
        "multi_hop_question_generation",
        "question_rewriting",
        "lighteval",
        "citation_score_filtering",
    ]:
        stage_cfg = getattr(pipeline, stage_name, None)
        if stage_cfg:
            is_enabled = stage_cfg.run if stage_cfg else False
            if is_enabled:
                desc = mapping.get(stage_name, stage_name.replace("_", " ").title())
                lines.append(f"- **{stage_name}**: {desc}")
    return "\n".join(lines)


# Helper function to extract settings without circular import
def _extract_settings_impl(config: Any):
    """Extract HF settings from config. Import here to avoid circular dependency."""
    from yourbench.utils.dataset_engine import _extract_settings

    return _extract_settings(config)


def _generate_and_upload_dataset_card(config: Any, template_path: str | None = None) -> None:
    """
    Internal implementation that generates and uploads a dataset card to Hugging Face Hub.

    This is the core implementation function called by the public upload_dataset_card() function.
    It handles the actual card generation and uploading without performing configuration checks.

    The dataset card includes:
    1. Pipeline subset descriptions based on enabled stages
    2. Full sanitized configuration for reproducibility
    3. YourBench version and other metadata
    4. Preserved dataset_info from the existing card for proper configuration display

    Args:
        config: Configuration dictionary containing HF settings
        template_path: Optional custom template path
    """
    logger.info("Starting dataset card upload process")

    if _is_offline():
        logger.warning("Offline mode enabled. Skipping dataset card upload.")
        return

    try:
        # Get dataset repo name
        settings = _extract_settings_impl(config)
        dataset_repo_name = settings.repo_id
        logger.info(f"Uploading card for dataset: {dataset_repo_name}")

        # Load template
        if not template_path:
            # Try to find template in utils directory
            current_dir = os.path.dirname(__file__)
            template_path = os.path.join(current_dir, "yourbench_card_template.md")

        logger.info(f"Loading template from: {template_path}")

        if not os.path.exists(template_path):
            logger.error(f"Template file not found: {template_path}")
            return

        with open(template_path, "r", encoding="utf-8") as f:
            template_str = f.read()

        logger.debug(f"Template loaded successfully, length: {len(template_str)} characters")

        # Get HF token
        token = settings.token

        # Extract dataset_info section from existing README if available
        config_data = extract_dataset_info(repo_id=dataset_repo_name, token=token)
        logger.info(f"Extracted dataset_info section, length: {len(config_data) if config_data else 0} characters")

        # Get hf_configuration (supports both attribute and dict access)
        hf_config = config.hf_configuration

        # Get pretty_name or generate from dataset name
        pretty_name = getattr(hf_config, "pretty_name", None)
        if not pretty_name:
            dataset_name = dataset_repo_name.split("/")[-1]
            pretty_name = dataset_name.replace("-", " ").replace("_", " ").title()

        card_data = DatasetCardData(pretty_name=pretty_name)
        logger.info(f"Created card data with pretty_name: {card_data.pretty_name}")

        # Get YourBench version
        from importlib.metadata import PackageNotFoundError, version

        try:
            version_str = version("yourbench")
        except PackageNotFoundError:
            version_str = "dev"

        # Get footer
        footer = getattr(hf_config, "footer", None)
        if not footer:
            footer = "*(This dataset card was automatically generated by YourBench)*"

        # Prepare template variables
        template_vars = {
            "pretty_name": card_data.pretty_name,
            "yourbench_version": version_str,
            "config_yaml": _serialize_config_for_card(config),
            "pipeline_subsets": _get_pipeline_subset_info(config),
            "config_data": config_data,
            "footer": footer,
        }

        logger.info("Rendering dataset card from template")
        logger.debug(f"Template variables: {list(template_vars.keys())}")

        # Render card with our template and variables
        card = DatasetCard.from_template(card_data=card_data, template_str=template_str, **template_vars)

        logger.info("Template rendered successfully")
        logger.debug(f"Rendered card content length: {len(str(card))} characters")

        # Push to hub
        logger.info(f"Pushing dataset card to hub: {dataset_repo_name}")
        card.push_to_hub(dataset_repo_name, token=token)

        logger.success(f"Dataset card successfully uploaded to: https://huggingface.co/datasets/{dataset_repo_name}")

    except Exception as e:
        logger.error(f"Failed to upload dataset card: {e}")
        logger.exception("Full traceback:")


def upload_dataset_card(config: Any) -> None:
    """
    Public interface to generate and upload a dataset card to Hugging Face Hub.

    This function performs configuration checks (like upload_card setting and offline mode)
    and then delegates to the internal _generate_and_upload_dataset_card() implementation.
    It should be called at the end of the pipeline when all subsets are available.

    Args:
        config: Pipeline configuration dictionary containing 'hf_configuration'
               with settings like 'upload_card' flag
    """
    try:
        # Check if card upload is enabled in config
        hf_config = config.hf_configuration
        upload_card = getattr(hf_config, "upload_card", True)
        if upload_card is None:
            upload_card = True

        if not upload_card:
            logger.info("Dataset card upload disabled in configuration. Skipping card upload.")
            return

        if _is_offline():
            logger.info("Offline mode enabled. Skipping dataset card upload.")
            return

        logger.info("Uploading dataset card with complete pipeline information")
        _generate_and_upload_dataset_card(config)

    except Exception as e:
        logger.error(f"Error uploading dataset card: {e}")
