import os
from typing import Any, Dict, Optional

from loguru import logger

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""

    pass


def _get_full_dataset_repo_name(config: Dict[str, Any]) -> str:
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

        # Prepend organization only if it exists and is not already part of the dataset name
        if organization and "/" not in dataset_name:
            dataset_name = f"{organization}/{dataset_name}"

        return dataset_name
    except ConfigurationError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise e


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
