from typing import Any, Dict, List

from datasets import Dataset, load_dataset
from loguru import logger

from yourbench.utils.dataset_engine import handle_dataset_push, make_dataset_name
from yourbench.utils.inference_engine import run_parallel_inference
from yourbench.utils.load_prompt import load_prompt
from yourbench.utils.parsing_engine import extract_content_from_xml_tags


class DatasetConstants:
    """Constants related to dataset operations."""

    TITLE_KEY = "document_name"
    CONTENT_KEY = "document_content"
    SUMMARY_KEY = "document_summary"
    SUMMARY_PROMPT_KEY = "general.summarize_document"
    FAILED_SUMMARY_STRING = "This document does not contain a summary."
    MIN_SUMMARY_LENGTH = 10


def prepare_summary_prompts(
    dataset: Dataset, prompt_template: str
) -> List[List[Dict[str, str]]]:
    """Prepares prompts for document summarization.

    Args:
        dataset: The dataset containing documents to summarize.
        prompt_template: The template string for generating summaries.

    Returns:
        List of formatted prompts ready for inference.
    """
    user_prompts = []
    logger.info("Preparing prompts for document summarization")

    for index, row in enumerate(dataset):
        document_text = f"Title: {row[DatasetConstants.TITLE_KEY]}\nContent: {row[DatasetConstants.CONTENT_KEY]}"
        templated_prompt = prompt_template.format(document=document_text)
        user_prompt = [{"role": "user", "content": templated_prompt}]
        user_prompts.append(user_prompt)

        if index > 0 and index % 100 == 0:
            logger.debug(f"Prepared prompts for {index} documents")

    return user_prompts


def validate_and_clean_summaries(summaries: List[str]) -> List[str]:
    """Validates generated summaries and replaces invalid ones with failure message.

    Args:
        summaries: List of generated summaries.

    Returns:
        List of validated and cleaned summaries.
    """
    validation_failures = 0
    cleaned_summaries = []

    for summary in summaries:
        if not summary or len(summary) < DatasetConstants.MIN_SUMMARY_LENGTH:
            cleaned_summaries.append(DatasetConstants.FAILED_SUMMARY_STRING)
            validation_failures += 1
        else:
            cleaned_summaries.append(summary)

    if validation_failures > 0:
        logger.warning(
            f"Found {validation_failures} failed summaries out of {len(summaries)} total documents"
        )

    return cleaned_summaries


def generate_summaries_for_documents(config: Dict[str, Any]) -> None:
    """Generates summaries for documents in a dataset using configured LLM.

    Args:
        config: Configuration dictionary containing all necessary parameters.

    Raises:
        Exception: If any step in the summary generation process fails.
    """
    logger.info("Starting summary generation process")
    dataset_name_key = config["pipeline"]["generate_summaries"]["source_dataset_name"]
    target_dataset_name_key = config["pipeline"]["generate_summaries"][
        "target_dataset_name"
    ]
    # Load dataset and prompt
    dataset = load_dataset(make_dataset_name(config, dataset_name_key), split="train")
    prompt_template = load_prompt(DatasetConstants.SUMMARY_PROMPT_KEY)

    # Prepare and run inference
    user_prompts = prepare_summary_prompts(dataset, prompt_template)
    logger.info(f"Starting parallel inference for {len(user_prompts)} documents")
    responses = run_parallel_inference(user_prompts, config)
    logger.success("Completed parallel inference")

    # Process responses
    logger.info("Extracting summaries from model responses")
    parsed_summaries = [
        extract_content_from_xml_tags(response, "final_summary")
        for response in responses
    ]

    # Validate and clean summaries
    cleaned_summaries = validate_and_clean_summaries(parsed_summaries)

    # Update dataset
    logger.info("Updating dataset with generated summaries")
    if DatasetConstants.SUMMARY_KEY in dataset.column_names:
        logger.debug("Removing existing summary column")
        dataset = dataset.remove_columns(DatasetConstants.SUMMARY_KEY)

    dataset = dataset.add_column(DatasetConstants.SUMMARY_KEY, cleaned_summaries)

    # Save results
    handle_dataset_push(config, target_dataset_name_key, dataset)
    logger.success("Summary generation process completed successfully")
