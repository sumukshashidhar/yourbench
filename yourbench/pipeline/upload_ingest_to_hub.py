# === CODEWRITING_GUIDELINES COMPLIANT ===
# Logging with loguru, graceful error handling, descriptive naming, 
# and thorough Google-style docstrings are used.

"""
Author: @sumukshashidhar

Module: upload_ingest_to_hub

Purpose:
    This module defines the `upload_ingest_to_hub` stage of the YourBench pipeline.
    In this stage, any markdown documents previously ingested (e.g., via the `ingestion` stage)
    can be packaged and uploaded to the Hugging Face Hub (or saved locally as a Hugging Face Dataset).
    The resulting dataset will contain a row per markdown file with standardized fields such as:
    - `document_id`
    - `document_text`
    - `document_filename`
    - `document_metadata`

Usage:
    1. Include or enable the `upload_ingest_to_hub` stage in your pipeline configuration:
        pipeline:
          upload_ingest_to_hub:
            run: true
            source_documents_dir: data/ingested/markdown
            # (Optional) override output_dataset_name, output_subset, etc.

    2. Ensure you have valid Hugging Face Hub credentials set in `hf_configuration.token` if you
       want to push to a private or protected dataset.
    3. Run the main pipeline (for example, via `yourbench.main.run_pipeline(config_file)`),
       and the code below will automatically execute when it reaches this stage.

Implementation Details:
    - The module locates all `.md` files in the configured source directory.
    - Each file is read, assigned a unique `document_id`, and stored in memory
      as an `IngestedDocument`.
    - These documents are converted to a Hugging Face Dataset object, which is then saved
      or pushed to the Hugging Face Hub using `save_dataset`.
    - Error handling logs warnings if no files are found, or if files are empty.
    - Logs and exceptions are written to a dedicated stage-level log file
      (via `loguru`), ensuring clarity for debugging or usage reports.
"""

import os
import glob
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from loguru import logger
from datasets import Dataset

from yourbench.utils.loading_engine import load_config
from yourbench.utils.dataset_engine import save_dataset

# Log to a stage-specific log file for thorough debugging information.
# The file rotates weekly to prevent excessive growth.
logger.add("logs/upload_ingest_to_hub.log", level="DEBUG", rotation="1 week")


@dataclass
class IngestedDocument:
    """
    Data model representing a single ingested Markdown document.

    Attributes:
        document_id (str):
            Unique ID for the document (typically a UUID4 string).
        document_text (str):
            Raw text content from the markdown file.
        document_filename (str):
            The original filename of the markdown file.
        document_metadata (Dict[str, Any]):
            Additional metadata, such as file size or arbitrary user-defined fields.
    """
    document_id: str
    document_text: str
    document_filename: str
    document_metadata: Dict[str, Any] = field(default_factory=dict)


def run(config: Dict[str, Any]) -> None:
    """
    Primary function to execute the 'upload_ingest_to_hub' stage.

    This function aggregates markdown documents from a given source directory
    (configured in `pipeline.upload_ingest_to_hub.source_documents_dir`) into a
    Hugging Face Dataset, which is then saved locally or pushed to the Hub.

    Args:
        config (Dict[str, Any]):
            The overall pipeline configuration dictionary. Relevant keys:

            - config["pipeline"]["upload_ingest_to_hub"]["run"] (bool):
                Whether to run this stage.
            - config["pipeline"]["upload_ingest_to_hub"]["source_documents_dir"] (str):
                Directory path for the ingested markdown files.
            - config["hf_configuration"]["token"] (str, optional):
                Hugging Face token for authentication if uploading a private dataset.
            - config["hf_configuration"]["private"] (bool):
                Whether to keep the dataset private on the Hub (defaults to True).
            - config["hf_configuration"]["global_dataset_name"] (str):
                Base dataset name on Hugging Face (can be overridden).
            - config["pipeline"]["upload_ingest_to_hub"]["output_dataset_name"] (str, optional):
                The name of the dataset to save to/push to on the Hugging Face Hub.
            - config["pipeline"]["upload_ingest_to_hub"]["output_subset"] (str, optional):
                Subset name for partial saving (default is this stage name).

    Raises:
        ValueError:
            If `source_documents_dir` is missing in the config, indicating incomplete config.
    """
    stage_name = "upload_ingest_to_hub"
    stage_cfg = config.get("pipeline", {}).get(stage_name, {})

    # Check if this stage is turned off in config
    if not stage_cfg.get("run", False):
        logger.info(f"Stage '{stage_name}' is disabled. Skipping.")
        return

    source_dir: Optional[str] = stage_cfg.get("source_documents_dir")
    if not source_dir:
        error_msg = (
            f"Missing required field 'source_documents_dir' in pipeline.{stage_name}. "
            f"Cannot proceed with uploading ingested documents."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Determine the final dataset name and subset from config
    hf_cfg = config.get("hf_configuration", {})
    output_dataset_name = stage_cfg.get("output_dataset_name", hf_cfg.get("global_dataset_name"))
    output_subset = stage_cfg.get("output_subset", stage_name)

    # Show key info about Hugging Face Hub config
    hf_token: Optional[str] = hf_cfg.get("token", None)
    hf_private: bool = hf_cfg.get("private", True)
    logger.info(f"Starting '{stage_name}' stage: uploading ingested files from '{source_dir}'")
    logger.debug(f"Hugging Face dataset name: '{output_dataset_name}' (private={hf_private})")

    # Collect .md files
    md_file_paths = glob.glob(os.path.join(source_dir, "*.md"))
    if not md_file_paths:
        logger.warning(f"No .md files found in '{source_dir}'. Stage will end with no output.")
        return

    # Read them into Python objects
    ingested_documents = _collect_markdown_files(md_file_paths)
    if not ingested_documents:
        logger.warning("No valid markdown documents found. No dataset to upload.")
        return

    # Convert the ingested markdown docs to a Hugging Face Dataset
    dataset = _convert_ingested_docs_to_dataset(ingested_documents)

    # Save or push the dataset to the configured location
    logger.info(f"Saving dataset to name='{output_dataset_name}', subset='{output_subset}'")
    save_dataset(
        dataset=dataset,
        step_name=stage_name,
        config=config,
        output_dataset_name=output_dataset_name,
        output_subset=output_subset
    )
    logger.success(f"Successfully completed '{stage_name}' stage.")


def _collect_markdown_files(md_file_paths: List[str]) -> List[IngestedDocument]:
    """
    Gather Markdown documents from the given file paths and store them in data classes.

    Args:
        md_file_paths (List[str]):
            A list of absolute/relative paths to `.md` files.

    Returns:
        List[IngestedDocument]:
            A list of `IngestedDocument` objects, one per valid markdown file discovered.

    Side Effects:
        Logs a warning for any unreadable or empty markdown files.
    """
    ingested_docs: List[IngestedDocument] = []
    for file_path in md_file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as file_handle:
                content = file_handle.read().strip()

            if not content:
                logger.warning(f"Skipping empty markdown file: {file_path}")
                continue

            doc_id = str(uuid.uuid4())
            ingested_docs.append(
                IngestedDocument(
                    document_id=doc_id,
                    document_text=content,
                    document_filename=os.path.basename(file_path),
                    document_metadata={"file_size": os.path.getsize(file_path)}
                )
            )
            logger.debug(f"Loaded markdown file: {file_path} (doc_id={doc_id})")

        except Exception as e:
            logger.error(
                f"Error reading file '{file_path}'. Skipping. Reason: {str(e)}"
            )

    return ingested_docs


def _convert_ingested_docs_to_dataset(ingested_docs: List[IngestedDocument]) -> Dataset:
    """
    Convert a list of ingested markdown documents into a Hugging Face Dataset object.

    Args:
        ingested_docs (List[IngestedDocument]):
            List of `IngestedDocument` objects to be packaged in a dataset.

    Returns:
        Dataset:
            A Hugging Face Dataset constructed from the provided documents,
            with columns: 'document_id', 'document_text', 'document_filename',
            and 'document_metadata'.
    """
    # Prepare data structure for Hugging Face Dataset
    records = {
        "document_id": [],
        "document_text": [],
        "document_filename": [],
        "document_metadata": [],
    }

    for doc in ingested_docs:
        records["document_id"].append(doc.document_id)
        records["document_text"].append(doc.document_text)
        records["document_filename"].append(doc.document_filename)
        records["document_metadata"].append(doc.document_metadata)

    dataset = Dataset.from_dict(records)
    logger.debug(f"Constructed HF Dataset with {len(dataset)} entries from ingested documents.")
    return dataset
