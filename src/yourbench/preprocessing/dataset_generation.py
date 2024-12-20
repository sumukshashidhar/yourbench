import uuid
from pathlib import Path
from typing import List, Tuple

from datasets import Dataset
from loguru import logger


def get_directory_name(config: dict) -> str:
    return config["selected_choices"]["generate_dataset"]["files_directory"]


def get_dataset_name(config: dict) -> str:
    return config["configurations"]["hf_organization"] + "/" + config["selected_choices"]["generate_dataset"]["dataset_name"]


def handle_dataset_push(dataset: Dataset, config: dict) -> None:
    dataset_name = get_dataset_name(config)

    if config["configurations"]["push_to_huggingface"]:
        privacy = False if config["configurations"]["set_hf_repo_visibility"] != "private" else True
        logger.info(f"Pushing dataset '{dataset_name}' to Hugging Face Hub (privacy={privacy})")
        try:
            dataset.push_to_hub(dataset_name, private=privacy)
            logger.success(f"Successfully pushed dataset to Hugging Face Hub: {dataset_name}")
        except Exception as error:
            logger.error(f"Failed to push dataset to Hugging Face Hub: {str(error)}")
            raise
    else:
        logger.info(f"Saving dataset locally to: {dataset_name}")
        dataset.save_to_disk(dataset_name)
        logger.success(f"Successfully saved dataset to disk: {dataset_name}")


def read_files(source_directory: str) -> List[Tuple[str, str, str]]:
    """
    Recursively reads files from source directory and returns a list of tuples containing:
    (filename, subdirectory path, file content)
    """
    logger.info(f"Starting to read files from directory: {source_directory}")
    files_data = []
    source_path = Path(source_directory)

    for file_path in source_path.rglob('*'):
        if file_path.is_file():
            logger.debug(f"Processing file: {file_path}")
            rel_path = file_path.relative_to(source_path)
            subdir_path = str(rel_path.parent) if str(rel_path.parent) != '.' else ''

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.debug(f"Successfully read file: {file_path}")
                files_data.append((
                    file_path.name.split(".")[0],
                    subdir_path,
                    content
                ))
            except Exception as error:
                logger.warning(f"Failed to read file {file_path}: {str(error)}")
                continue

    logger.info(f"Completed reading {len(files_data)} files")
    return files_data


def generate_dataset(config: dict):
    logger.info("Starting dataset generation process")
    directory_name = get_directory_name(config)
    logger.debug(f"Reading files from directory: {directory_name}")

    files_data = read_files(directory_name)
    logger.info(f"Converting {len(files_data)} files to dataset format")

    files_data_dict = [
        {
            "id": str(uuid.uuid4()),
            "title": file_data[0],
            "category": file_data[1],
            "content": file_data[2]
        }
        for file_data in files_data
    ]

    logger.debug("Creating Hugging Face Dataset object")
    dataset = Dataset.from_list(files_data_dict)
    logger.info(f"Created dataset with {len(dataset)} entries")

    handle_dataset_push(dataset, config)
    logger.success("Dataset generation completed successfully")
    return
