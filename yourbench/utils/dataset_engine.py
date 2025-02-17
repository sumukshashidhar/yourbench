from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from loguru import logger


def make_dataset_name(config: dict, dataset_name: str) -> str:
    if "/" in dataset_name:
        logger.warning("Dataset name already contains a slash. Returning original dataset name.")
        return dataset_name
    logger.debug("Dataset name does not contain a slash. Constructing fully qualified dataset name.")
    organization = config["configurations"]["huggingface"]["hf_organization"]
    return f"{organization}/{dataset_name}"


def handle_dataset_push(config: dict, dataset_name: str, dataset: Dataset) -> None:
    """Handles the process of saving or pushing a dataset to Hugging Face Hub, with optional concatenation."""

    push_to_hf = config.get("configurations", {}).get("huggingface", {}).get("push_to_huggingface", False)
    visibility = config.get("configurations", {}).get("huggingface", {}).get("set_hf_repo_visibility", "private")
    concat = config.get("configurations", {}).get("huggingface", {}).get("concat_if_exists", False)  # concat flag

    fully_qualified_dataset_name = make_dataset_name(config, dataset_name)
    logger.info(
        f"Processing dataset '{fully_qualified_dataset_name}' with {dataset.num_rows} rows and columns: {dataset.column_names}"
    )

    if not push_to_hf:
        save_dataset_locally(dataset, fully_qualified_dataset_name)
        return

    private = visibility != "public"
    logger.info(f"Handling dataset push to Hugging Face Hub (private={private})")

    existing_dataset = load_existing_ds(fully_qualified_dataset_name)

    if concat and existing_dataset:
        dataset = concatenate_ds(dataset, existing_dataset)
        if dataset is None:
            return  # Error logged inside the function

    elif existing_dataset is not None:
        delete_existing_ds(fully_qualified_dataset_name)

    push_ds_to_hub(dataset, fully_qualified_dataset_name, private)


def save_dataset_locally(dataset: Dataset, dataset_name: str) -> None:
    """Saves the dataset to local disk."""
    logger.info(f"Saving dataset locally to: {dataset_name}")
    dataset.save_to_disk(dataset_name)
    logger.success(f"Successfully saved dataset to disk: {dataset_name}")


def load_existing_ds(dataset_name: str) -> Dataset | None:
    """Attempts to load an existing dataset from Hugging Face Hub, if it exists."""
    try:
        existing_dataset = load_dataset(dataset_name, split="train")
        logger.info(f"Existing dataset found: '{dataset_name}' with {existing_dataset.num_rows} rows.")
        return existing_dataset
    except FileNotFoundError:
        logger.info(f"Dataset '{dataset_name}' does not exist. It will be created.")
        return None
    except Exception as e:
        logger.error(f"Error while checking for existing dataset: {e}")
        return None


def concatenate_ds(new_dataset: Dataset, existing_dataset: Dataset) -> Dataset | None:
    """Concatenates datasets if they have matching features."""
    if new_dataset.features != existing_dataset.features:
        logger.error("The new dataset and the existing dataset have different features. Cannot concatenate.")
        return None

    try:
        combined_dataset = concatenate_datasets([existing_dataset, new_dataset])
        logger.warning(f"Datasets concatenated successfully. The new dataset has {combined_dataset.num_rows} rows.")
        return combined_dataset
    except Exception as e:
        logger.error(f"Error while concatenating datasets: {e}")
        return None


def delete_existing_ds(dataset_name: str) -> None:
    """Deletes an existing dataset from Hugging Face Hub only if it exists."""
    try:
        api = HfApi()

        # Check if the dataset exists
        existing_datasets = api.list_datasets(author=dataset_name.split("/")[0])
        if not any(ds.id == dataset_name for ds in existing_datasets):
            logger.warning(f"Dataset '{dataset_name}' does not exist, skipping deletion.")
            return

        logger.warning(f"Deleting existing dataset '{dataset_name}' before overwriting.")
        api.delete_repo(repo_id=dataset_name, repo_type="dataset")

    except Exception as e:
        logger.error(f"Failed to delete existing dataset: {e}")


def push_ds_to_hub(dataset: Dataset, dataset_name: str, private: bool) -> None:
    """Pushes the dataset to the Hugging Face Hub."""
    try:
        dataset.push_to_hub(dataset_name, private=private)
        logger.success(f"Successfully pushed dataset to Hugging Face Hub: {dataset_name}")
    except Exception as e:
        logger.error(f"An error occurred while pushing the dataset: {e}")
