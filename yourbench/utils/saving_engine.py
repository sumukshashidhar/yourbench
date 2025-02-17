from datasets import Dataset, load_dataset, concatenate_datasets
from typing import Dict, Any
def save_dataset(dataset: Dataset, step_name: str, config: Dict[str, Any], output_dataset_name: str) -> None:
    """
    Save a dataset to huggingface
    """
    dataset.save_to_disk(config["pipeline"][step_name]["local_dataset_path"])
    # check if we need to concatenate with an existing dataset
    if config["pipeline"][step_name]["concat_existing_dataset"]:
        existing_dataset = load_dataset(config["pipeline"][step_name]["output_dataset_name"])
        dataset = concatenate_datasets([existing_dataset, dataset])
    # push to hub
    dataset.push_to_hub(output_dataset_name, token=config["hf_configuration"]["token"], private=config["hf_configuration"]["private"])
    return