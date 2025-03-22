# summarization.py
# =============================================================================
# Author: @sumukshashidhar
#
# Module: Summarization Pipeline Stage
# =============================================================================
"""
Summarization Stage
===================

This module handles the summarization stage of the YourBench pipeline. It takes
documents (with their raw text) and generates concise yet comprehensive summaries
for each document.

Usage:
------
1. Ensure the pipeline configuration has an entry for the `summarization` stage
   with the desired settings. For example:

   summarization:
     run: true
     timeout_seconds: 300

2. When the pipeline runs, it loads the target dataset, calls the summarization
   model(s) to produce summaries, logs intermediate steps, and saves the updated
   dataset with new columns:
     - raw_document_summary
     - document_summary
     - summarization_model

Error Handling & Logging:
-------------------------
- All errors are logged using `loguru` to `logs/summarization.log`.
- The stage attempts to proceed with partial data even if some calls fail, never
  abruptly terminating the pipeline.

Important Notes:
----------------
- This stage relies on the `run_inference` utility function from yourbench.utils.inference_engine
  for concurrency, timeouts, and model management.
- Summaries are extracted from the model's output by parsing <final_summary> XML tags.
- If no valid summary is found, the pipeline substitutes a fallback string.

See Also:
---------
- yourbench.utils.inference_engine for concurrency logic
- yourbench.utils.dataset_engine for loading/saving dataset
"""

from typing import Any

from datasets import Dataset
from loguru import logger

from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.inference_engine import InferenceCall, run_inference
from yourbench.utils.parsing_engine import extract_content_from_xml_tags
from yourbench.utils.prompts import SUMMARIZATION_USER_PROMPT


def duplicate_rows(
    dataset: dict[str, Any], num_duplicates: int = 1
) -> dict[str, list[Any]]:
    """
    Create a dictionary that repeats each value in the dataset multiple times.

    Args:
        dataset (dict[str, Any]): A dictionary representing dataset columns.
        num_duplicates (int): How many times to duplicate each row.

    Returns:
        dict[str, list[Any]]: A new dictionary where each key's list is repeated
        num_duplicates times.
    """
    repeated_data = {}
    for key, value in dataset.items():
        repeated_data[key] = [val for val in value for _ in range(num_duplicates)]
    return repeated_data


def _prepare_inference_calls(dataset: Dataset) -> list[InferenceCall]:
    """
    Create InferenceCall objects for each document to be summarized.

    Args:
        dataset (Dataset): The dataset containing the documents to be summarized.

    Returns:
        list[InferenceCall]: A list of inference calls to be sent to the summarization model.
    """
    documents: list[str]
    try:
        documents = dataset["document_text"]
    except KeyError:
        logger.error("Dataset does not contain 'document_text' column. Cannot proceed.")
        return []
    except Exception as exc:
        logger.error("Unexpected error reading 'document_text': {}", str(exc))
        return []

    inference_calls: list[InferenceCall] = []
    for doc_text in documents:
        user_msg_content = SUMMARIZATION_USER_PROMPT.format(document=doc_text)
        user_msg = {"role": "user", "content": user_msg_content}
        inference_calls.append(
            InferenceCall(messages=[user_msg], tags=["summarization"])
        )

    logger.info("Prepared {} inference calls for summarization.", len(inference_calls))
    return inference_calls


def _perform_summarization_inference(
    config: dict[str, Any], inference_calls: list[InferenceCall]
) -> dict[str, list[str]] | None:
    """
    Perform the actual inference (summarization) calls to the model.

    Args:
        config (dict[str, Any]): The entire pipeline configuration.
        inference_calls (list[InferenceCall]): The inference calls to be processed.

    Returns:
        dict[str, list[str]] | None: A dictionary of model_name -> list of summaries,
                                     or None if inference fails.
    """
    if not inference_calls:
        logger.error("No inference calls were prepared; skipping summarization.")
        return None

    response_dict = run_inference(
        config=config,
        step_name="summarization",
        inference_calls=inference_calls,
    )

    if response_dict is None:
        logger.error("Inference for summarization returned no data.")
        return None
    return response_dict


def _extract_summaries_from_model_output(
    dataset: Dataset, response_dict: dict[str, list[str]]
) -> tuple[str, list[str], list[str]]:
    """
    Take the raw inference responses, parse out the <final_summary>, and
    return the model name and final summaries.

    Args:
        dataset (Dataset): The original dataset (to compare lengths).
        response_dict (dict[str, list[str]]): Inference responses keyed by model name.

    Returns:
        tuple[str, list[str], list[str]]:
            - The name of the summarization model,
            - The list of raw model summaries,
            - The list of extracted/parsed final summaries.
    """
    documents = dataset["document_text"]
    try:
        # Typically only one model is used, so take the first key
        summ_model_name = list(response_dict.keys())[0]
        model_raw_summaries: list[str] = response_dict.get(summ_model_name, [])

        if not model_raw_summaries:
            logger.error(
                "Model '{}' returned no summaries. Check your model configuration.",
                summ_model_name,
            )
            return "", [], []
    except IndexError:
        logger.error("No valid model keys found in the response dictionary.")
        return "", [], []

    # Ensure there's a 1:1 match between documents and summaries
    if len(model_raw_summaries) != len(documents):
        logger.warning(
            "Mismatch in number of summaries vs documents. Adjusting list size."
        )
        while len(model_raw_summaries) < len(documents):
            model_raw_summaries.append("")
        if len(model_raw_summaries) > len(documents):
            model_raw_summaries = model_raw_summaries[: len(documents)]

    extracted_summaries: list[str] = []
    for i, raw_resp in enumerate(model_raw_summaries):
        logger.debug("Parsing doc index {}, raw response length={}", i, len(raw_resp))
        try:
            parsed = extract_content_from_xml_tags(raw_resp, "final_summary")
        except Exception as parse_exc:
            logger.error("Error parsing doc index {}: {}", i, str(parse_exc))
            parsed = ""

        parsed_stripped = parsed.strip()
        if not parsed_stripped:
            logger.warning("No <final_summary> content found for doc index {}.", i)
            extracted_summaries.append("No summary available for this document.")
        else:
            extracted_summaries.append(parsed_stripped)

    return summ_model_name, model_raw_summaries, extracted_summaries


def _add_summary_columns_to_dataset(
    dataset: Dataset,
    summ_model_name: str,
    raw_summaries: list[str],
    extracted_summaries: list[str],
) -> Dataset:
    """
    Add the raw and extracted summaries (and model name) as columns to the dataset.

    Args:
        dataset (Dataset): The original dataset.
        summ_model_name (str): The summarization model's name.
        raw_summaries (list[str]): The unprocessed summaries from the model.
        extracted_summaries (list[str]): The parsed/extracted final summaries.

    Returns:
        Dataset: The dataset with the added columns.
    """
    try:
        dataset = dataset.add_column("raw_document_summary", raw_summaries)
    except Exception as e:
        logger.error("Error adding 'raw_document_summary': {}", str(e))

    try:
        dataset = dataset.add_column("document_summary", extracted_summaries)
    except Exception as e:
        logger.error("Error adding 'document_summary': {}", str(e))

    try:
        dataset = dataset.add_column(
            "summarization_model", [summ_model_name] * len(dataset)
        )
    except Exception as e:
        logger.error("Error adding 'summarization_model': {}", str(e))

    return dataset


def run(config: dict[str, Any]) -> None:
    """
    Execute the Summarization Stage of YourBench.

    This stage:
      1. Loads a dataset of documents from the configuration.
      2. Uses one or more summarization models to generate summaries for each doc.
      3. Attempts to parse each model's output for <final_summary> tags.
      4. Logs results and saves updated columns in the dataset.

    Args:
        config (dict[str, Any]): The entire pipeline configuration dictionary.

    Returns:
        None. The function saves the resulting dataset to disk/HF Hub if successful.
    """
    stage_cfg = config.get("pipeline", {}).get("summarization", {})
    if not stage_cfg.get("run", False):
        logger.info("Summarization stage is disabled. Skipping.")
        return

    logger.info("Beginning Summarization Stage...")

    # 1) Load dataset
    dataset = custom_load_dataset(config=config, subset="ingested")
    logger.info(f"Loaded ingested subset with {len(dataset)} rows for summarization.")

    # 2) Prepare calls to summarization model
    inference_calls = _prepare_inference_calls(dataset)
    if not inference_calls:
        # Already logged errors if the dataset is invalid
        return

    # 3) Perform summarization
    response_dict = _perform_summarization_inference(config, inference_calls)
    if not response_dict:
        return

    # 4) Gather and parse model responses
    summ_model_name, model_raw_summaries, extracted_summaries = (
        _extract_summaries_from_model_output(dataset, response_dict)
    )
    if not summ_model_name:
        return

    # 5) Add new columns to the dataset
    dataset = _add_summary_columns_to_dataset(
        dataset, summ_model_name, model_raw_summaries, extracted_summaries
    )

    # 6) Save updated dataset
    custom_save_dataset(dataset=dataset, config=config, subset="summarized")
    logger.success("Summarization stage completed successfully.")
