from loguru import logger
from datasets import Dataset, load_dataset, concatenate_datasets


def make_dataset_name(config: dict, dataset_name: str) -> str:
    """
    Make a dataset name from the config and the dataset name.
    """
    # check if the dataset name is already in the format of hf_organization/dataset_name, to avoid double slashes
    if "/" in dataset_name:
        logger.warning(
            "Dataset name already contains a slash. Returning original dataset name."
        )
        return dataset_name
    logger.debug(
        "Dataset name does not contain a slash. Returning formatted dataset name."
    )
    return (
        config["configurations"]["huggingface"]["hf_organization"] + "/" + dataset_name
    )


def handle_dataset_push(config: dict, dataset_name: str, dataset: Dataset) -> None:
    push_key = config["configurations"]["huggingface"]["push_to_huggingface"]
    visibility_key = config["configurations"]["huggingface"]["set_hf_repo_visibility"]
    concat_if_exists = config["configurations"]["huggingface"]["concat_if_exists"]

    dataset_name = make_dataset_name(config, dataset_name)
    # first check if we're meant to just save to disk
    if not push_key:
        logger.info(f"Saving dataset locally to: {dataset_name}")
        dataset.save_to_disk(dataset_name)
        logger.success(f"Successfully saved dataset to disk: {dataset_name}")
        return

    # otherwise, we need to push to hf
    # let us figure out the privacy first. default is private for safety
    if visibility_key == "public":
        privacy = False
    else:
        privacy = True

    logger.info(
        f"Pushing dataset '{dataset_name}' to Hugging Face Hub (privacy={privacy})"
    )
    # let us try to load the dataset if it already exists
    try:
        existing_dataset = load_dataset(dataset_name, split="train")
        # this means we found the dataset
        if concat_if_exists:
            logger.info(
                f"Concatenating existing dataset with new dataset: {dataset_name}"
            )
            dataset = concatenate_datasets([existing_dataset, dataset])
        else:
            # we just push the new dataset
            dataset.push_to_hub(dataset_name, private=privacy)
            logger.success(
                f"Successfully pushed dataset to Hugging Face Hub: {dataset_name}"
            )
            return
    except Exception:
        # this means the dataset does not exist
        logger.info(f"Dataset does not exist. Pushing new dataset: {dataset_name}")
        dataset.push_to_hub(dataset_name, private=privacy)
        logger.success(
            f"Successfully pushed dataset to Hugging Face Hub: {dataset_name}"
        )
