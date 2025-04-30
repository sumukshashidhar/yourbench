import os
from typing import Any, Dict, Optional

from loguru import logger

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from huggingface_hub import HfApi, whoami
from huggingface_hub.utils import HFValidationError


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""

    pass

def _safe_get_organization(config: Dict, dataset_name: str, organization: str, token: str) -> str:
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
                logger.warning(
                    f"Invalid Hugging Face token provided: {ve}. Proceeding without organization prefix."
                )
                organization = None
            except Exception as e:  # Catch other potential issues like network errors
                logger.warning(
                    f"Failed to fetch username via whoami: {e}. Proceeding without organization prefix."
                )
                organization = None
        else:
            logger.warning(
                "'hf_organization' not set or expanded, and no 'token' provided in config. Proceeding without organization prefix."
            )
            organization = None  # Ensure organization is None if logic falls through
    return organization


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
        token = hf_config.get("token")

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
    Load a dataset subset from Hugging Face, ensuring that we handle subsets correctly.
    """
    dataset_repo_name = _get_full_dataset_repo_name(config)

    # TODO: add an optional loading from a local path
    logger.info(f"Loading dataset HuggingFace Hub with repo_id='{dataset_repo_name}'")
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
    """

    dataset_repo_name = _get_full_dataset_repo_name(config)

    local_dataset_dir = config.get("local_dataset_dir", None)
    if local_dataset_dir and save_local:
        logger.info(f"Saving dataset localy to: '{local_dataset_dir}'")
        if subset:
            local_dataset = DatasetDict({subset: dataset})
            local_dataset_dir = os.path.join(local_dataset_dir, subset)
        else:
            local_dataset = dataset

        os.makedirs(local_dataset_dir, exist_ok=True)
        local_dataset.save_to_disk(local_dataset_dir)
        logger.success(f"Dataset successfully saved localy to: '{local_dataset_dir}'")

    if config["hf_configuration"].get("concat_if_exist", False):
        existing_dataset = custom_load_dataset(config=config, subset=subset)
        dataset = concatenate_datasets([existing_dataset, dataset])
        logger.info("Concatenated dataset with an existing one")

    if subset:
        config_name = subset
    else:
        config_name = "default"

    if push_to_hub:
        logger.info(f"Pushing dataset to HuggingFace Hub with repo_id='{dataset_repo_name}'")
        dataset.push_to_hub(
            repo_id=dataset_repo_name,
            private=config["hf_configuration"].get("private", True),
            config_name=config_name,
        )
        logger.success(f"Dataset successfully pushed to HuggingFace Hub with repo_id='{dataset_repo_name}'")
