import os
from typing import Any, Dict, Optional

from loguru import logger

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from huggingface_hub import HfApi, whoami, DatasetCardData, DatasetCard
from huggingface_hub.utils import HFValidationError


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""

    pass


def _is_offline_mode() -> bool:
    """Check if offline mode is enabled via environment variable."""
    return os.environ.get("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes")


def _safe_get_organization(config: Dict, dataset_name: str, organization: str, token: str) -> str:
    # In offline mode, don't try to fetch organization
    if _is_offline_mode():
        logger.info("Offline mode detected. Skipping organization fetch.")
        return organization

    if not organization or (isinstance(organization, str) and organization.startswith("$")):
        if isinstance(organization, str) and organization.startswith("$"):
            # Log if it was explicitly set but unexpanded
            var_name = organization[1:].split("/")[0]
            logger.warning(
                f"Environment variable '{var_name}' used in 'hf_organization' ('{organization}') is not set or expanded."
            )

        if token:
            logger.info(
                "'hf_organization' not set or expanded, attempting to fetch default username using provided token."
            )
            try:
                user_info = whoami(token=token)
                default_username = user_info.get("name")
                if default_username:
                    organization = default_username
                    logger.info(f"Using fetched default username '{organization}' as the organization.")
                else:
                    logger.warning(
                        "Could not retrieve username from token information. Proceeding without organization prefix."
                    )
                    organization = None
            except HFValidationError as ve:
                logger.warning(f"Invalid Hugging Face token provided: {ve}. Proceeding without organization prefix.")
                organization = None
            except Exception as e:  # Catch other potential issues like network errors
                logger.warning(f"Failed to fetch username via whoami: {e}. Proceeding without organization prefix.")
                organization = None
        else:
            logger.warning(
                "'hf_organization' not set or expanded, and no 'token' provided in config. Proceeding without organization prefix."
            )
            organization = None  # Ensure organization is None if logic falls through
    return organization


def extract_readme_metadata(repo_id: str, token: Optional[str] = None) -> str:
    """Extracts the metadata from the README.md file of the dataset repository.
    We have to download the previous README.md file in the repo, extract the metadata from it.
    Args:
        repo_id: The ID of the repository to push to, from the `push_to_hub` method.
        token: The token to authenticate with the Hugging Face Hub, from the `push_to_hub` method.
    Returns:
        The metadata extracted from the README.md file of the dataset repository as a str.
    """
    try:
        from pathlib import Path
        import re
        from huggingface_hub.file_download import hf_hub_download

        readme_path = Path(
            hf_hub_download(repo_id, "README.md", repo_type="dataset", token=token)
        )
        # Extract the content between the '---' markers
        metadata_match = re.findall(r"---\n(.*?)\n---", readme_path.read_text(), re.DOTALL)

        if not metadata_match:
            logger.debug("No YAML metadata found in the README.md")
            return ""

        return metadata_match[0]

    except Exception as e:
        logger.debug(f"Failed to extract metadata from README.md: {e}")
        return ""


def extract_dataset_info(repo_id: str, token: Optional[str] = None) -> str:
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


def _serialize_config_for_card(config: Dict[str, Any]) -> str:
    """
    Sanitize and serialize pipeline config to YAML for inclusion in dataset card.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config serialization")
    from copy import deepcopy

    def _sanitize(obj, key=None):
        if isinstance(obj, dict):
            return {k: _sanitize(v, k) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, str):
            # Keep placeholders
            if obj.startswith("$"):
                return obj
            # Mask only api_key arguments
            if key and "api_key" in key.lower():
                return "$API_KEY"
            # Mask OpenAI API keys
            if obj.startswith("sk-"):
                return "$OPENAI_API_KEY"
            # Mask HuggingFace tokens
            if obj.startswith("hf_"):
                return "$HF_TOKEN"
            return obj
        # Explicitly return boolean, integer, float, and None values unchanged
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj

    sanitized = _sanitize(deepcopy(config))
    return yaml.safe_dump(sanitized, sort_keys=False, default_flow_style=False)


def _get_pipeline_subset_info(config: Dict[str, Any]) -> str:
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
    pipeline = config.get("pipeline", {})
    lines = []
    for stage, cfg in pipeline.items():
        if isinstance(cfg, dict) and cfg.get("run"):
            desc = mapping.get(stage, stage.replace("_", " ").title())
            lines.append(f"- **{stage}**: {desc}")
    return "\n".join(lines)


def _get_full_dataset_repo_name(config: Dict[str, Any]) -> str:
    """
    Determines the full Hugging Face dataset repository name.

    If 'hf_organization' is not provided or refers to an unexpanded environment
    variable, it attempts to infer the username using the provided 'hf_token'.
    If 'hf_dataset_name' refers to an unexpanded environment variable, it raises
    an error.

    Args:
        config (Dict[str, Any]): The loaded configuration dictionary.

    Returns:
        str: The full dataset repository name (e.g., 'username/dataset_name' or 'dataset_name').

    Raises:
        ConfigurationError: If required configuration keys are missing, if
                            'hf_dataset_name' is unexpanded, or if the final
                            repo ID is invalid.
    """
    try:
        if "hf_configuration" not in config:
            error_msg = "Missing 'hf_configuration' in config"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)

        hf_config = config["hf_configuration"]
        if "hf_dataset_name" not in hf_config:
            error_msg = "Missing 'hf_dataset_name' in hf_configuration"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)

        dataset_name = hf_config["hf_dataset_name"]
        organization = hf_config.get("hf_organization")
        token = hf_config.get("token") if "token" in hf_config else os.getenv("HF_TOKEN", None)

        # Attempt to get default username if organization is missing or unexpanded
        organization = _safe_get_organization(config, dataset_name, organization, token)

        # Dataset name MUST be expanded correctly
        if isinstance(dataset_name, str) and dataset_name.startswith("$"):
            var_name = dataset_name[1:].split("/")[0]
            error_msg = (
                f"Environment variable '{var_name}' used in required 'hf_dataset_name' ('{dataset_name}') is not set or expanded. "
                f"Please set the '{var_name}' environment variable or update the configuration."
            )
            logger.error(error_msg)
            raise ConfigurationError(error_msg)

        # Construct the full name
        full_dataset_name = dataset_name
        if organization and "/" not in dataset_name:
            full_dataset_name = f"{organization}/{dataset_name}"

        # Skip Hub validation in offline mode
        if _is_offline_mode():
            logger.debug(f"Offline mode detected. Skipping Hub validation for repo ID '{full_dataset_name}'")
            return full_dataset_name

        # Use HfApi for robust validation
        api = HfApi()
        try:
            api.repo_info(repo_id=full_dataset_name, repo_type="dataset", token=token)
            # If repo exists, validation passed implicitly (though repo_info might fetch info we don't strictly need here)
            logger.debug(
                f"Repo ID '{full_dataset_name}' seems valid (checked via repo_info). Existing status not determined here."
            )
        except HFValidationError as ve:
            # This catches validation errors during repo_info call if the name format is wrong
            error_msg = f"Constructed Hugging Face repo ID '{full_dataset_name}' is invalid: {ve}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from ve
        except Exception as e:
            # Handle cases where repo doesn't exist (which is fine) vs other errors
            # Note: repo_info raises RepositoryNotFoundError subclass of HfHubHTTPError
            # We only care about *validation* here, not existence. If it gets past HFValidationError, assume format is okay.
            # Other exceptions might indicate network issues etc. but not invalid ID format per se.
            # We'll let push_to_hub handle non-existence later if needed.
            if "404" in str(e) or "Repository Not Found" in str(e):
                logger.debug(
                    f"Repo ID '{full_dataset_name}' format appears valid, but repository does not exist (or access denied). This is acceptable for creation."
                )
            else:
                # Log unexpected errors during validation check but don't necessarily block
                logger.warning(
                    f"Unexpected issue during repo ID validation check for '{full_dataset_name}': {e}. Proceeding, but push/load might fail."
                )

        return full_dataset_name

    except ConfigurationError:  # Re-raise config errors directly
        raise
    except Exception as e:  # Catch unexpected errors
        logger.exception(f"Unexpected error in _get_full_dataset_repo_name: {e}")
        raise ConfigurationError(f"Failed to determine dataset repo name: {e}") from e


def custom_load_dataset(config: Dict[str, Any], subset: Optional[str] = None) -> Dataset:
    """
    Load a dataset subset from a local directory if specified, otherwise from Hugging Face.
    In offline mode, only load from local directory.
    """
    local_dataset_dir = config.get("local_dataset_dir", None)
    if (
        local_dataset_dir is None
        and "hf_configuration" in config
        and "local_dataset_dir" in config["hf_configuration"]
    ):
        local_dataset_dir = config["hf_configuration"].get("local_dataset_dir")

    # First try loading from local path
    if local_dataset_dir:
        if os.path.exists(local_dataset_dir):
            logger.info(f"Loading dataset locally from '{local_dataset_dir}'")
            dataset = load_from_disk(local_dataset_dir)
            # If subset is specified and this is a DatasetDict, return only the subset
            if subset and isinstance(dataset, DatasetDict):
                if subset in dataset:
                    return dataset[subset]
                else:
                    logger.warning(f"Subset '{subset}' not found in local dataset. Returning empty dataset.")
                    return Dataset.from_dict({})
            return dataset
        else:
            logger.warning(f"local_dataset_dir '{local_dataset_dir}' does not exist.")
            if _is_offline_mode():
                raise ValueError("Offline mode is enabled but local dataset not found")
            else:
                logger.warning("Falling back to Hugging Face Hub.")

    # If we're in offline mode and made it here, the local dataset doesn't exist
    if _is_offline_mode():
        logger.warning("Offline mode enabled but no local dataset found. Returning empty dataset.")
        return Dataset.from_dict({})

    # If we're here, try to get from Hub
    dataset_repo_name = _get_full_dataset_repo_name(config)
    logger.info(f"Loading dataset from HuggingFace Hub with repo_id='{dataset_repo_name}'")

    # If subset name does NOT exist, return an empty dataset to avoid the crash:
    try:
        return load_dataset(dataset_repo_name, name=subset, split="train")
    except ValueError as e:
        # If the config was not found, we create an empty dataset
        if "BuilderConfig" in str(e) and "not found" in str(e):
            logger.warning(f"No existing subset '{subset}'. Returning empty dataset.")
            return Dataset.from_dict({})
        else:
            raise


def custom_save_dataset(
    dataset: Dataset,
    config: Dict[str, Any],
    subset: Optional[str] = None,
    save_local: bool = True,
    push_to_hub: bool = True,
) -> None:
    """
    Save a dataset subset locally and push it to Hugging Face Hub.

    When saving locally:
    - If a subset is specified, it will be added to an existing dataset or
      create a new DatasetDict containing that subset.
    - All subsets are saved to the same local_dataset_dir.
    """
    # In offline mode, force save local and disable push to hub
    if _is_offline_mode():
        save_local = True
        if push_to_hub:
            logger.warning("Offline mode enabled. Disabling push_to_hub operation.")
            push_to_hub = False

    dataset_repo_name = _get_full_dataset_repo_name(config)

    local_dataset_dir = config.get("local_dataset_dir", None)
    if (
        local_dataset_dir is None
        and "hf_configuration" in config
        and "local_dataset_dir" in config["hf_configuration"]
    ):
        local_dataset_dir = config["hf_configuration"].get("local_dataset_dir")

    if local_dataset_dir and save_local:
        logger.info(f"Saving dataset locally to: '{local_dataset_dir}'")

        # Check if dataset exists at the specified location
        if os.path.exists(local_dataset_dir):
            try:
                # Try to load existing dataset
                existing_dataset = load_from_disk(local_dataset_dir)
                if subset:
                    if isinstance(existing_dataset, DatasetDict):
                        # To avoid the "dataset can't overwrite itself" error,
                        # create a new dataset dictionary instead of modifying the existing one
                        new_dataset_dict = DatasetDict()

                        # Copy all existing subsets except the one we're updating
                        for key, value in existing_dataset.items():
                            if key != subset:
                                new_dataset_dict[key] = value

                        # Add the new subset
                        new_dataset_dict[subset] = dataset
                        logger.info(f"Adding/updating subset '{subset}' to existing dataset")
                        local_dataset = new_dataset_dict
                    else:
                        # Existing dataset is not a DatasetDict, convert it
                        logger.info("Converting existing dataset to DatasetDict to add subset")
                        if subset == "default" or subset == "train":
                            # If existing dataset is the default subset, convert to DatasetDict
                            local_dataset = DatasetDict({"default": existing_dataset, subset: dataset})
                        else:
                            # If existing dataset is not a DatasetDict, convert it to one with "default" as the key
                            local_dataset = DatasetDict({"default": existing_dataset, subset: dataset})
                else:
                    # No subset specified, simply overwrite the existing dataset
                    local_dataset = dataset
            except Exception as e:
                # If there was an error loading the existing dataset
                logger.warning(f"Error loading existing dataset: {e}. Creating a new dataset.")
                if subset:
                    local_dataset = DatasetDict({subset: dataset})
                else:
                    local_dataset = dataset
        else:
            # No existing dataset, create a new one
            if subset:
                local_dataset = DatasetDict({subset: dataset})
            else:
                local_dataset = dataset

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(local_dataset_dir), exist_ok=True)

        try:
            # Save the dataset to disk
            local_dataset.save_to_disk(local_dataset_dir)
            logger.success(f"Dataset successfully saved locally to: '{local_dataset_dir}'")
        except PermissionError as e:
            if "dataset can't overwrite itself" in str(e):
                # Handle the specific error where a dataset can't overwrite itself
                logger.warning("Dataset can't overwrite itself. Attempting to save with a temporary directory...")
                import shutil
                import tempfile

                # Create a temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save to temporary directory first
                    local_dataset.save_to_disk(temp_dir)

                    # Remove the existing dataset directory
                    shutil.rmtree(local_dataset_dir)

                    # Copy from temporary directory to the target directory
                    shutil.copytree(temp_dir, local_dataset_dir)

                logger.success(
                    f"Dataset successfully saved locally to: '{local_dataset_dir}' using a temporary directory"
                )
            else:
                # Re-raise if it's a different permission error
                raise

    if config["hf_configuration"].get("concat_if_exist", False) and not _is_offline_mode():
        existing_dataset = custom_load_dataset(config=config, subset=subset)
        dataset = concatenate_datasets([existing_dataset, dataset])
        logger.info("Concatenated dataset with an existing one")

    if subset:
        config_name = subset
    else:
        config_name = "default"

    if push_to_hub and not _is_offline_mode():
        logger.info(f"Pushing dataset to HuggingFace Hub with repo_id='{dataset_repo_name}'")
        dataset.push_to_hub(
            repo_id=dataset_repo_name,
            private=config["hf_configuration"].get("private", True),
            config_name=config_name,
        )
        logger.success(f"Dataset successfully pushed to HuggingFace Hub with repo_id='{dataset_repo_name}'")


def replace_dataset_columns(
    dataset: Dataset, columns_data: dict[str, list], preserve_metadata: bool = False
) -> Dataset:
    """
    Replace columns in a dataset by removing existing columns and adding new ones.

    This helper function handles the common pattern of:
    1. Removing existing columns (if they exist)
    2. Adding new columns with computed data

    Args:
        dataset: The input dataset to modify
        columns_data: Dictionary mapping column names to their data lists
        preserve_metadata: If True, attempts to preserve column metadata (not implemented)

    Returns:
        Updated dataset with replaced columns

    Note:
        Column metadata (types, features) is not preserved in the current implementation.
        New columns will have types inferred from the provided data.
    """
    # Remove existing columns to prevent duplication errors
    columns_to_replace = list(columns_data.keys())
    existing_columns_to_remove = [col for col in columns_to_replace if col in dataset.column_names]

    if existing_columns_to_remove:
        logger.info(f"Removing existing columns before adding new ones: {existing_columns_to_remove}")
        dataset = dataset.remove_columns(existing_columns_to_remove)

    # Add new columns
    for column_name, column_data in columns_data.items():
        dataset = dataset.add_column(column_name, column_data)

    return dataset


def _generate_and_upload_dataset_card(
    config: Dict[str, Any], 
    template_path: str | None = None
) -> None:
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
    
    if _is_offline_mode():
        logger.warning("Offline mode enabled. Skipping dataset card upload.")
        return
    
    try:
        # Get dataset repo name
        dataset_repo_name = _get_full_dataset_repo_name(config)
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
        hf_config = config.get("hf_configuration", {})
        token = hf_config.get("token") or os.getenv("HF_TOKEN", None)
        
        # Extract dataset_info section from existing README if available
        config_data = extract_dataset_info(repo_id=dataset_repo_name, token=token)
        logger.info(f"Extracted dataset_info section, length: {len(config_data) if config_data else 0} characters")
        
        # Use explicitly configured pretty_name or generate one from the dataset name
        if "pretty_name" in hf_config:
            pretty_name = hf_config["pretty_name"]
        else:
            dataset_name = dataset_repo_name.split("/")[-1]
            pretty_name = dataset_name.replace("-", " ").replace("_", " ").title()
            
        card_data_kwargs = {
            "pretty_name": pretty_name
        }
        
        # Create DatasetCardData with our metadata
        card_data = DatasetCardData(**card_data_kwargs)
        logger.info(f"Created card data with pretty_name: {card_data.pretty_name}")
        
        # Get YourBench version
        from importlib.metadata import version, PackageNotFoundError
        
        try:
            version_str = version("yourbench")
        except PackageNotFoundError:
            # Fallback for development installs
            version_str = "dev"
        
        # Prepare template variables
        template_vars = {
            "pretty_name": card_data.pretty_name,
            "yourbench_version": version_str,
            "config_yaml": _serialize_config_for_card(config),
            "pipeline_subsets": _get_pipeline_subset_info(config),
            "config_data": config_data,  # Use the extracted dataset_info section
            "footer": hf_config.get("footer", "*(This dataset card was automatically generated by YourBench)*")
        }
        
        logger.info("Rendering dataset card from template")
        logger.debug(f"Template variables: {list(template_vars.keys())}")
        
        # Render card with our template and variables 
        card = DatasetCard.from_template(
            card_data=card_data,
            template_str=template_str,
            **template_vars
        )
        
        logger.info("Template rendered successfully")
        logger.debug(f"Rendered card content length: {len(str(card))} characters")
        
        # Push to hub
        logger.info(f"Pushing dataset card to hub: {dataset_repo_name}")
        card.push_to_hub(dataset_repo_name, token=token)
        
        logger.success(f"Dataset card successfully uploaded to: https://huggingface.co/datasets/{dataset_repo_name}")
        
    except Exception as e:
        logger.error(f"Failed to upload dataset card: {e}")
        logger.exception("Full traceback:")


def upload_dataset_card(config: Dict[str, Any]) -> None:
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
        hf_config = config.get("hf_configuration", {})
        upload_card = hf_config.get("upload_card", True)
        
        if not upload_card:
            logger.info("Dataset card upload disabled in configuration. Skipping card upload.")
            return
            
        if _is_offline_mode():
            logger.info("Offline mode enabled. Skipping dataset card upload.")
            return
            
        logger.info("Uploading dataset card with complete pipeline information")
        _generate_and_upload_dataset_card(config)
        
    except Exception as e:
        logger.error(f"Error uploading dataset card: {e}")