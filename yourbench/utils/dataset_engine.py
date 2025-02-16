from datasets import load_dataset, Dataset
from typing import Dict, Any

def smart_load_dataset(dataset_name: str, config: Dict[str, Any]) -> Dataset:
    """
    Load a dataset from huggingface, with the option to concatenate with an existing dataset
    """
    # check the name, does it have a organization in it?
    if "/" not in dataset_name:
        dataset_name = f"{config['hf_configuration']['hf_organization']}/{dataset_name}"
    # load the dataset
    dataset = load_dataset(dataset_name, token=config["hf_configuration"]["token"], split="train")
    return dataset
