import uuid
from pathlib import Path
from typing import List, Tuple

from datasets import Dataset


def read_files(source_directory: str) -> List[Tuple[str, str, str]]:
    """
    Recursively reads files from source directory and returns a list of tuples containing:
    (filename, subdirectory path, file content)
    """
    files_data = []
    source_path = Path(source_directory)

    for file_path in source_path.rglob('*'):
        if file_path.is_file():
            # Get relative path parts excluding filename
            rel_path = file_path.relative_to(source_path)
            subdir_path = str(rel_path.parent) if str(rel_path.parent) != '.' else ''

            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                files_data.append((
                    file_path.name.split(".")[0],           # filename
                    subdir_path,              # subdirectory path
                    content                   # file content
                ))
            except Exception as _:
                continue

    return files_data


def generate_dataset(config: dict):
    files_data = read_files(config["datasets"]["source_directory"])
    # make this a dictionary
    files_data_dict = [
        {
            "id": str(uuid.uuid4()),
            "title": file_data[0],
            "category": file_data[1],
            "content": file_data[2]
        }
        for file_data in files_data
    ]
    # convert the dict to a huggingface dataset
    dataset = Dataset.from_list(files_data_dict)
    dataset.push_to_hub(config["datasets"]["document_dataset_name"])
    return
