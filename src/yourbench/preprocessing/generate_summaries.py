from datasets import load_dataset

from yourbench.utils.inference_engine import get_batch_completion
from yourbench.utils.load_prompt import load_prompt
from yourbench.utils.parsing_engine import extract_content_from_xml_tags


TITLE_KEY = "title"
CONTENT_KEY = "content"
SUMMARY_PROMPT_KEY = "general.summarize_document"
FAILED_SUMMARY_STRING = "This document does not contain a summary."


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
    if TITLE_KEY not in dataset.column_names:
        raise ValueError(f"Dataset must contain a '{TITLE_KEY}' column")
    if CONTENT_KEY not in dataset.column_names:
        raise ValueError(f"Dataset must contain a '{CONTENT_KEY}' column")

    return dataset


def generate_summaries_for_documents(hf_dataset_name: str, config: dict):
    """
    Given a huggingface dataset in a compatible format, we generate summaries using the
    given LLM configurations.

    Args:
        hf_dataset_name (str): The name of the huggingface dataset to generate summaries for.

    """
    dataset = _validate_dataset(hf_dataset_name)
    created_dataset_name = (
        f"{hf_dataset_name}-w-summaries"
        if "document_with_summary_dataset_name" not in config["datasets"]
        else config["datasets"]["document_with_summary_dataset_name"]
    )
    prompt = load_prompt(SUMMARY_PROMPT_KEY)
    user_prompts = []
    for row in dataset:
        templated_prompt = prompt.format(
            document=f"Title: {row[TITLE_KEY]}\nContent: {row[CONTENT_KEY]}"
        )
        user_prompt = [{"role": "user", "content": templated_prompt}]
        user_prompts.append(user_prompt)

    # check if the model_config/summarization_model is defined
    if "summarization_model" in config["model_config"]:
        summarization_model = config["model_config"]["summarization_model"]
    else:
        # choose model_0 as the summarization model
        summarization_model = config["model_config"]["model_0"]

    responses = get_batch_completion(
        model_name=summarization_model["model_name"],
        model_type=summarization_model["model_type"],
        user_prompts=user_prompts,
        batch_size=2,
    )

    parsed_responses = [
        extract_content_from_xml_tags(response, "final_summary")
        for response in responses
    ]

    for i in range(len(parsed_responses)):
        if not parsed_responses[i] or len(parsed_responses[i]) < 10:
            parsed_responses[i] = FAILED_SUMMARY_STRING

    # Update or create the summary column
    if "summary" in dataset.column_names:
        dataset = dataset.remove_columns("summary")
    dataset = dataset.add_column("summary", parsed_responses)
    try:
        dataset.push_to_hub(created_dataset_name)
    except Exception as push_error:
        dataset.save_to_disk(created_dataset_name)
        print(f"Failed to push dataset to hub: {push_error}")


if __name__ == "__main__":
    generate_summaries_for_documents("sumuks/fairytales")
