import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import Dataset
from loguru import logger
from utils.dataset_engine import handle_dataset_push


def get_directory_name(config: Dict[str, Any]) -> str:
    """Retrieves the files directory name from the configuration.

    Args:
        config: Configuration dictionary containing pipeline settings.

    Returns:
        str: Directory name where source files are located.
    """
    try:
        return config["pipeline"]["generate_dataset"]["files_directory"]
    except KeyError as e:
        logger.error(f"Missing required configuration key: {e}")
        raise


def read_files(source_directory: str) -> List[Tuple[str, str, str]]:
    """Recursively reads files from source directory.

    Args:
        source_directory: Path to the source directory containing files to process.

    Returns:
        List of tuples containing (filename, subdirectory path, file content).

    Raises:
        FileNotFoundError: If the source directory doesn't exist.
    """
    logger.info(f"Starting to read files from directory: {source_directory}")
    files_data = []
    source_path = Path(source_directory)

    if not source_path.exists():
        logger.error(f"Source directory not found: {source_directory}")
        raise FileNotFoundError(f"Directory does not exist: {source_directory}")

    total_files = 0
    processed_files = 0

    for file_path in source_path.rglob("*"):
        if file_path.is_file():
            total_files += 1
            logger.debug(f"Processing file: {file_path}")
            rel_path = file_path.relative_to(source_path)
            subdir_path = str(rel_path.parent) if str(rel_path.parent) != "." else ""

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                processed_files += 1
                logger.debug(f"Successfully read file: {file_path}")
                files_data.append((file_path.name.split(".")[0], subdir_path, content))
            except Exception as error:
                logger.warning(f"Failed to read file {file_path}: {str(error)}")
                continue

    logger.info(f"Completed reading {processed_files}/{total_files} files successfully")
    return files_data


def generate_dataset(config: Dict[str, Any]) -> None:
    """Generates a dataset from files in the specified directory.

    Args:
        config: Configuration dictionary containing pipeline settings.

    Raises:
        KeyError: If required configuration keys are missing.
    """
    logger.info("Starting dataset generation process")

    try:
        directory_name = get_directory_name(config)
        dataset_name = config["pipeline"]["generate_dataset"]["dataset_name"]
    except KeyError as e:
        logger.error(f"Missing required configuration key: {e}")
        raise

    logger.debug(f"Reading files from directory: {directory_name}")
    files_data = read_files(directory_name)
    logger.info(f"Converting {len(files_data)} files to dataset format")

    files_data_dict = [
        {
            "document_id": str(uuid.uuid4()),
            "document_name": file_data[0],
            "document_category": file_data[1],
            "document_content": file_data[2],
        }
        for file_data in files_data
    ]

    logger.debug("Creating Hugging Face Dataset object")
    dataset = Dataset.from_list(files_data_dict)
    logger.info(f"Created dataset with {len(dataset)} entries")

    handle_dataset_push(config, dataset_name, dataset)
    logger.success("Dataset generation completed successfully")
