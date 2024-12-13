from datasets import Dataset, load_dataset
from loguru import logger

from yourbench.utils.inference_engine import run_parallel_inference
from yourbench.utils.load_prompt import load_prompt
from yourbench.utils.parsing_engine import extract_content_from_xml_tags


TITLE_KEY = "title"
CONTENT_KEY = "content"
SUMMARY_PROMPT_KEY = "general.summarize_document"
FAILED_SUMMARY_STRING = "This document does not contain a summary."


def handle_dataset_push(dataset: Dataset, dataset_name: str, config: dict) -> None:

    if config["configurations"]["push_to_huggingface"]:
        privacy = False if config["configurations"]["set_hf_repo_visibility"] != "private" else True
        logger.info(f"Pushing dataset '{dataset_name}' to Hugging Face Hub (privacy={privacy})")
        try:
            dataset.push_to_hub(config["configurations"]["hf_organization"] + "/" + dataset_name, private=privacy)
            logger.success(f"Successfully pushed dataset to Hugging Face Hub: {dataset_name}")
        except Exception as error:
            logger.error(f"Failed to push dataset to Hugging Face Hub: {str(error)}")
            raise
    else:
        logger.info(f"Saving dataset locally to: {dataset_name}")
        dataset.save_to_disk(dataset_name)
        logger.success(f"Successfully saved dataset to disk: {dataset_name}")


def _load_dataset(config: dict):
    organization = config["configurations"]["hf_organization"]
    dataset_name = config["selected_choices"]["generate_summaries"]["document_dataset_name"]
    logger.debug(f"Loading dataset from Hugging Face: {organization}/{dataset_name}")
    dataset = load_dataset(f"{organization}/{dataset_name}", split="train")
    logger.debug(f"Successfully loaded dataset with {len(dataset)} entries")
    return dataset


def generate_summaries_for_documents(config: dict):
    """
    Given a huggingface dataset in a compatible format, we generate summaries using the
    given LLM configurations.

    Args:
        hf_dataset_name (str): The name of the huggingface dataset to generate summaries for.

    """
    logger.info("Starting summary generation process")

    # load the dataset
    dataset = _load_dataset(config)
    created_dataset_name = config["selected_choices"]["generate_summaries"]["summary_dataset_name"]

    logger.debug("Loading summary generation prompt template")
    prompt = load_prompt(SUMMARY_PROMPT_KEY)

    logger.info("Preparing prompts for document summarization")
    user_prompts = []
    for index, row in enumerate(dataset):
        templated_prompt = prompt.format(
            document=f"Title: {row[TITLE_KEY]}\nContent: {row[CONTENT_KEY]}"
        )
        user_prompt = [{"role": "user", "content": templated_prompt}]
        user_prompts.append(user_prompt)
        if index % 100 == 0 and index > 0:
            logger.debug(f"Prepared prompts for {index} documents")

    logger.info(f"Starting parallel inference for {len(user_prompts)} documents")
    responses = run_parallel_inference(user_prompts, config)
    logger.success("Completed parallel inference")

    logger.info("Extracting summaries from model responses")
    parsed_responses = [
        extract_content_from_xml_tags(response, "final_summary")
        for response in responses
    ]

    logger.debug("Validating generated summaries")
    validation_failures = 0
    for i in range(len(parsed_responses)):
        if not parsed_responses[i] or len(parsed_responses[i]) < 10:
            parsed_responses[i] = FAILED_SUMMARY_STRING
            validation_failures += 1

    if validation_failures > 0:
        logger.warning(f"Found {validation_failures} failed summaries out of {len(parsed_responses)} total documents")

    logger.info("Updating dataset with generated summaries")
    if "summary" in dataset.column_names:
        logger.debug("Removing existing summary column")
        dataset = dataset.remove_columns("summary")
    dataset = dataset.add_column("summary", parsed_responses)

    handle_dataset_push(dataset, created_dataset_name, config)
    logger.success("Summary generation process completed successfully")
