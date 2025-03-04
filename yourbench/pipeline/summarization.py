from random import uniform
from datasets import Dataset
from typing import Dict, Any, List

from yourbench.utils.inference_engine import run_inference, InferenceCall
from yourbench.utils.prompts import SUMMARIZATION_USER_PROMPT
from yourbench.utils.saving_engine import save_dataset
from yourbench.utils.dataset_engine import smart_load_dataset
from yourbench.utils.parsing_engine import extract_content_from_xml_tags
from loguru import logger

def duplicate_rows(dataset, num_duplicates=20):
    # Return the example as a list repeated num_duplicates times
    return {k: [v] * num_duplicates for k, v in dataset.items()}

def run(config: Dict[str, Any]) -> None:
    """
    Run the summarization stage of the pipeline in parallel.

    Steps:
      1. Load the dataset specified by 'source_documents_dataset_name'.
      2. Create an InferenceCall for each document text.
      3. Pass all InferenceCall objects at once to 'run_inference' to allow parallel processing.
      4. Parse the raw model responses, extract <final_summary> content, and store them in the dataset.
      5. Save the updated dataset to local disk and optionally push to a Hugging Face Hub dataset.
    """
    # Check if the pipeline stage is active
    summary_cfg = config["pipeline"]["summarization"]
    if not summary_cfg.get("run", False):
        logger.info("Summarization stage disabled in config. Skipping.")
        return

    logger.info("Running summarization stage.")

    # 1. Load the dataset
    dataset = smart_load_dataset(summary_cfg["source_dataset_name"], config)
    logger.info("Loaded dataset with %d documents.", len(dataset))

    dataset = dataset.map(
        duplicate_rows,
        batched=False,
        remove_columns=dataset.column_names
    )

    # Flatten the dataset
    dataset = {k: sum(v, []) for k, v in dataset.to_dict().items()}
    dataset = Dataset.from_dict(dataset)

    # 2. Build the inference calls
    documents: List[str] = dataset["document_text"]

    inference_calls = []

    for document in documents:
        # Generate a random temperature in the range [0.7, 1.0]
        temperature_value = round(uniform(0.7, 1.0), 2)

        # Log the temperature for debugging
        logger.info(f"Selected temperature for document: {temperature_value}")

        # Create inference call
        call = InferenceCall(
            messages=[
                {
                    "role": "user",
                    "content": SUMMARIZATION_USER_PROMPT.format(document=document)
                }
            ],
            temperature=temperature_value
        )

        inference_calls.append(call)

    # 3. Run inference in parallel
    logger.info("Sending %d summarization calls to inference engine.", len(inference_calls))
    responses_dict = run_inference(config, "summarization", inference_calls)

    # We assume a single summarization model in model_roles["summarization"][0]
    summ_model = config["model_roles"]["summarization"][0]
    raw_summaries = responses_dict[summ_model]

    # 4. Parse out <final_summary> from each raw model response
    final_summaries = [
        extract_content_from_xml_tags(r, "final_summary") if r else ""
        for r in raw_summaries
    ]

    # Attach columns to the dataset
    dataset = dataset.add_column("raw_document_summary", raw_summaries)
    dataset = dataset.add_column("document_summary", final_summaries)
    dataset = dataset.add_column("summarization_model", [summ_model] * len(dataset))

    # 5. Save the updated dataset
    save_dataset(
        dataset,
        "summarization",
        config,
        summary_cfg["output_dataset_name"]
    )
    logger.success("Summarization stage completed successfully.")