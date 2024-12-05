from datasets import load_dataset
from yourbench.utils.load_prompt import load_prompt


def _validate_dataset(hf_dataset_name: str):
    """
    Validate the dataset exists, and is compatible for generating summaries. Datasets
    must contain a 'title' and 'content' column.

    Args:
        hf_dataset_name (str): The name of the huggingface dataset to validate.

    Raises:
        ValueError: If the dataset does not exist, or does not contain the required columns.
    """
    try:
        dataset = load_dataset(hf_dataset_name, split="train")
    except Exception as e:
        raise ValueError(f"Dataset {hf_dataset_name} not found") from e

    # check if the required columns are present
    if "title" not in dataset.column_names:
        raise ValueError("Dataset must contain a 'title' column")
    if "content" not in dataset.column_names:
        raise ValueError("Dataset must contain a 'content' column")

    return dataset


def generate_summaries_for_documents(hf_dataset_name: str):
    """
    Given a huggingface dataset in a compatible format, we generate summaries using the
    given LLM configurations.

    Args:
        hf_dataset_name (str): The name of the huggingface dataset to generate summaries for.

    """
    dataset = _validate_dataset(hf_dataset_name)
    prompt = load_prompt("general.summarize_document")


if __name__ == "__main__":
    generate_summaries_for_documents("cnn_dailymail")
