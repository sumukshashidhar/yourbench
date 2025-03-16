from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Dict, Any

def smart_get_source_dataset_name(stage_name: str, config: Dict[str, Any]) -> str:
    return config.get("pipeline", {}).get(stage_name, {}).get("source_dataset_name", config.get("hf_configuration", {}).get("global_dataset_name"))

def smart_get_source_subset(stage_name: str, config: Dict[str, Any]) -> str:
    return config.get("pipeline", {}).get(stage_name, {}).get("source_subset", "")

def smart_get_output_dataset_name(stage_name: str, config: Dict[str, Any]) -> str:
    return config.get("pipeline", {}).get(stage_name, {}).get("output_dataset_name", config.get("hf_configuration", {}).get("global_dataset_name"))

def smart_get_output_subset(stage_name: str, config: Dict[str, Any]) -> str:
    return config.get("pipeline", {}).get(stage_name, {}).get("output_subset", "")


def smart_load_dataset(dataset_name: str, config: Dict[str, Any], dataset_subset: str = "", split: str = "train") -> Dataset:
    """
    Load a dataset from huggingface, with the option to concatenate with an existing dataset
    """
    if not dataset_name:
        # try to get the global dataset name
        dataset_name = config.get("hf_configuration", {}).get("global_dataset_name")
    # check the name, does it have a organization in it?
    if "/" not in dataset_name:
        dataset_name = f"{config.get('hf_configuration', {}).get('hf_organization')}/{dataset_name}"
    # load the dataset
    dataset = load_dataset(
        dataset_name, token=config.get("hf_configuration", {}).get("token"), name=dataset_subset, split=split
        )
    return dataset

def save_dataset(dataset: Dataset, step_name: str, config: Dict[str, Any], output_dataset_name: str, output_subset: str = None, split: str = "train") -> None:
    """
    Save a dataset to huggingface
    """
    output_subset = output_subset or smart_get_output_subset(step_name, config)
    
    if not output_dataset_name:
        output_dataset_name = smart_get_output_dataset_name(step_name, config)
    local_path = config.get("pipeline", {}).get(step_name, {}).get("local_dataset_path", False)
    if local_path:
        dataset.save_to_disk(local_path)

    # check if we need to concatenate with an existing dataset
    if config.get("pipeline", {}).get(step_name, {}).get("concat_existing_dataset", False):
        try:
            existing_dataset = smart_load_dataset(output_dataset_name, config, output_subset, split)
            dataset = concatenate_datasets([existing_dataset, dataset])
        except Exception as e:
            logger.warning(f"Failed to concatenate existing dataset: {e}. Skipping concatenation.")
    # push to hub
    dataset.push_to_hub(output_dataset_name, token=config.get("hf_configuration", {}).get("token"), private=config.get("hf_configuration", {}).get("private"), split=split, config_name=output_subset)
    return