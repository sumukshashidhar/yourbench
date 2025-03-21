import os
from typing import Any, Dict, Optional
from datasets import Dataset, concatenate_datasets, load_dataset, DatasetDict
from loguru import logger


def _get_full_dataset_repo_name(config: Dict[str, Any]):
    dataset_name = config["hf_configuration"]["hf_dataset_name"]
    if "/" not in dataset_name:
        dataset_name = f"{config['hf_configuration']['hf_organization']}/{dataset_name}"

    return dataset_name


def custom_load_dataset(
    config: Dict[str, Any], subset: Optional[str] = None
) -> Dataset:
    """
    Load a dataset subset from Hugging Face, ensuring that we handle subsets correctly.
    """
    dataset_repo_name = _get_full_dataset_repo_name(config)

    # if no step given, load full dataset
    logger.info(f"Loading dataset HuggingFace Hub with repo_id='{dataset_repo_name}'")
    if subset:
        dataset = load_dataset(dataset_repo_name, name=subset, split="train")
    else:
        dataset = load_dataset(dataset_repo_name, split="train")

    return dataset


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
        logger.info(f"Concatenated dataset with an existing one")

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
