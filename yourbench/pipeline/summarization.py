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
     source_dataset_name: yourbench_dataset
     source_subset: ingested_documents
     output_dataset_name: yourbench_dataset
     output_subset: summarized_documents
     output_split: train

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

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional

from datasets import Dataset
from loguru import logger

from yourbench.utils.dataset_engine import save_dataset, smart_load_dataset
from yourbench.utils.inference_engine import InferenceCall, run_inference
from yourbench.utils.parsing_engine import extract_content_from_xml_tags
from yourbench.utils.prompts import SUMMARIZATION_USER_PROMPT

def _run_inference_with_timeout(
    config: Dict[str, Any], inference_calls: List[InferenceCall], stage_name: str, timeout_seconds: float
) -> Optional[Dict[str, List[str]]]:
    """
    Run inference with a forced timeout, preventing infinite hang.

    Args:
        config (Dict[str, Any]): Pipeline configuration dictionary.
        inference_calls (List[InferenceCall]): A list of calls to be passed to run_inference.
        stage_name (str): The pipeline stage name (e.g. "summarization").
        timeout_seconds (float): Timeout in seconds before we consider it a failure.

    Returns:
        Optional[Dict[str, List[str]]]: Dictionary of responses per model. If
        timed out or errored, returns None.
    """
    logger.info("Attempting inference with a maximum timeout of {} seconds...", timeout_seconds)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_inference, config, stage_name, inference_calls)
        try:
            result = future.result(timeout=timeout_seconds)
            if result is None or not isinstance(result, dict):
                logger.error("Inference returned None or invalid result type: {}", type(result))
                return None

            if len(result) == 0:
                logger.error("Inference returned an empty dictionary")
                return None

            for model_name, responses in result.items():
                if not responses:
                    logger.warning("Model '{}' returned empty response list", model_name)
                else:
                    logger.info("Received {} responses from model '{}'", len(responses), model_name)

            return result

        except FuturesTimeoutError:
            logger.error("Inference timed out after {} seconds.", timeout_seconds)
        except Exception as exc:
            logger.error("Error during inference: {}", str(exc))

    return None


def duplicate_rows(dataset: Dict[str, Any], num_duplicates: int = 1) -> Dict[str, List[Any]]:
    """
    Create a dictionary that repeats each value in the dataset multiple times.

    Args:
        dataset (Dict[str, Any]): A dictionary representing dataset columns.
        num_duplicates (int): How many times to duplicate each row.

    Returns:
        Dict[str, List[Any]]: A new dictionary where each key's list is repeated
        num_duplicates times.
    """
    repeated_data = {}
    for key, value in dataset.items():
        repeated_data[key] = [val for val in value for _ in range(num_duplicates)]
    return repeated_data


def run(config: Dict[str, Any]) -> None:
    """
    Execute the Summarization Stage of YourBench.

    This stage:
      1. Loads a dataset of documents from the configuration.
      2. Uses one or more summarization models to generate summaries for each doc.
      3. Attempts to parse each model's output for <final_summary> tags.
      4. Logs results and saves updated columns in the dataset.

    Args:
        config (Dict[str, Any]): The entire pipeline configuration dictionary.

    Returns:
        None. The function saves the resulting dataset to disk/HF Hub if successful.
    """
    stage_cfg = config.get("pipeline", {}).get("summarization", {})
    debug_mode: bool = config.get("settings", {}).get("debug", False)
    if not stage_cfg.get("run", False):
        logger.info("Summarization stage is disabled. Skipping.")
        return

    logger.info("Beginning Summarization Stage...")

    # 1) Load dataset
    source_dataset_name = stage_cfg.get(
        "source_dataset_name", config.get("hf_configuration", {}).get("global_dataset_name")
    )
    source_subset = stage_cfg.get("source_subset", "ingested_documents")
    try:
        dataset: Dataset = smart_load_dataset(source_dataset_name, config, dataset_subset=source_subset)
        logger.info("Loaded dataset '{}' with {} documents for summarization.", source_dataset_name, len(dataset))
    except Exception as exc:
        logger.error(
            "Failed to load dataset '{}': {}. Summarization stage cannot proceed.", source_dataset_name, str(exc)
        )
        return

    # 2) Prepare calls to summarization model
    try:
        documents: List[str] = dataset["document_text"]
    except KeyError:
        logger.error("Dataset does not contain 'document_text' column. Cannot proceed.")
        return
    except Exception as exc:
        logger.error("Unexpected error reading 'document_text': {}", str(exc))
        return

    inference_calls: List[InferenceCall] = []
    for idx, doc_text in enumerate(documents):
        user_msg_content = SUMMARIZATION_USER_PROMPT.format(document=doc_text)
        user_msg = {"role": "user", "content": user_msg_content}
        inference_calls.append(InferenceCall(messages=[user_msg], tags=["summarization"]))

    logger.info("Prepared {} inference calls for summarization.", len(inference_calls))

    # 3) Perform summarization with timeout
    timeout_seconds: float = stage_cfg.get("timeout_seconds", 1800.0)
    response_dict = _run_inference_with_timeout(
        config=config, inference_calls=inference_calls, stage_name="summarization", timeout_seconds=timeout_seconds
    )
    if response_dict is None:
        logger.error("Inference for summarization returned no data.")
        return

    if not response_dict:
        logger.error(
            "Inference returned an empty dictionary. This could indicate a configuration issue with models for the summarization stage."
        )
        return

    # 4) Gather model responses
    try:
        summ_model_name = list(response_dict.keys())[0]
        model_raw_summaries: List[str] = response_dict.get(summ_model_name, [])

        if not model_raw_summaries:
            logger.error("Model '{}' returned no summaries. Check your model configuration.", summ_model_name)
            return

    except IndexError:
        logger.error("No valid model keys found in the response dictionary.")
        return

    if len(model_raw_summaries) != len(documents):
        logger.warning("Mismatch in number of summaries vs documents. Adjusting list size.")
        while len(model_raw_summaries) < len(documents):
            model_raw_summaries.append("")
        if len(model_raw_summaries) > len(documents):
            model_raw_summaries = model_raw_summaries[: len(documents)]

    extracted_summaries: List[str] = []
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

    # 5) Add new columns to the dataset
    try:
        dataset = dataset.add_column("raw_document_summary", model_raw_summaries)
    except Exception as e:
        logger.error("Error adding 'raw_document_summary': {}", str(e))

    try:
        dataset = dataset.add_column("document_summary", extracted_summaries)
    except Exception as e:
        logger.error("Error adding 'document_summary': {}", str(e))

    try:
        dataset = dataset.add_column("summarization_model", [summ_model_name] * len(dataset))
    except Exception as e:
        logger.error("Error adding 'summarization_model': {}", str(e))

    # 6) Save updated dataset
    output_dataset_name = stage_cfg.get(
        "output_dataset_name", config.get("hf_configuration", {}).get("global_dataset_name")
    )
    output_subset = stage_cfg.get("output_subset", "summarized_documents")
    output_split = stage_cfg.get("output_split", "train")

    try:
        save_dataset(
            dataset=dataset,
            step_name="summarization",
            config=config,
            output_dataset_name=output_dataset_name,
            output_subset=output_subset,
            split=output_split,
        )
        logger.success("Summarization stage completed successfully.")
    except Exception as e:
        logger.error("Error saving summarized dataset: {}", str(e))
        raise e
