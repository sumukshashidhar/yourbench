from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Dict, Any

def smart_load_dataset(dataset_name: str, config: Dict[str, Any], split: str = "train") -> Dataset:
    """
    Load a dataset from huggingface, with the option to concatenate with an existing dataset
    """
    # check the name, does it have a organization in it?
    if "/" not in dataset_name:
        dataset_name = f"{config['hf_configuration']['hf_organization']}/{dataset_name}"
    # load the dataset
    dataset = load_dataset(dataset_name, token=config["hf_configuration"]["token"], split=split)
    return dataset

def save_dataset(dataset: Dataset, step_name: str, config: Dict[str, Any], output_dataset_name: str) -> None:
    """
    Save a dataset to huggingface
    """
    local_path = config["pipeline"][step_name].get("local_dataset_path")
    if local_path:
        dataset.save_to_disk(local_path)
    # check if we need to concatenate with an existing dataset
    if config["pipeline"][step_name]["concat_existing_dataset"]:
        existing_dataset = load_dataset(config["pipeline"][step_name]["output_dataset_name"])
        dataset = concatenate_datasets([existing_dataset, dataset])
    # push to hub
    dataset.push_to_hub(output_dataset_name, token=config["hf_configuration"]["token"], private=config["hf_configuration"]["private"])
    return