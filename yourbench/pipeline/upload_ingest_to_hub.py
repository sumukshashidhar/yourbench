# =============================================================================
# upload_ingest_to_hub.py
# =============================================================================
"""
Author: @sumukshashidhar

This module handles uploading ingested Markdown files to the Hugging Face Hub
as a dataset. It reads all Markdown files from a specified directory, converts
each file into an `IngestedDocument` data structure, and constructs a Hugging
Face `Dataset`. The resulting dataset is then saved (and optionally pushed) to
the Hub or to local storage, according to configuration.

Usage:
    - This module is typically invoked by the YourBench pipeline handler when
      the pipeline stage `upload_ingest_to_hub` is enabled in the config.
    - It requires a `source_documents_dir` from which to read all Markdown
      files and produce a dataset.

Key Steps:
    1. Read `.md` files from `source_documents_dir`.
    2. Create `IngestedDocument` objects containing file content and metadata.
    3. Convert them into a Hugging Face `Dataset`.
    4. Save/push the dataset to local storage or the Hugging Face Hub based on
       pipeline configurations in `config["hf_configuration"]`.

Example:
    pipeline:
      upload_ingest_to_hub:
        run: true
        source_documents_dir: "data/example/processed/ingested"
        output_subset: "ingested_documents"
        # Optionally specify dataset name, local path, etc.

"""


import os
import glob
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from loguru import logger
from datasets import Dataset

from yourbench.utils.loading_engine import load_config
from yourbench.utils.dataset_engine import save_dataset


@dataclass
class IngestedDocument:
    """
    Represents an ingested Markdown document.

    Attributes:
        document_id (str): Unique identifier for the document (UUID-based).
        document_text (str): The raw text content extracted from the markdown file.
        document_filename (str): The name of the source markdown file.
        document_metadata (Dict[str, Any]): Arbitrary metadata related to the document,
            such as file size, creation date, etc.
    """
    document_id: str
    document_text: str
    document_filename: str
    document_metadata: Dict[str, Any] = field(default_factory=dict)


def run(config: Dict[str, Any]) -> None:
    """
    Main entry point for the 'upload_ingest_to_hub' pipeline stage.

    This function reads configuration data to locate markdown files, converts them
    into a dataset, and saves/pushes the dataset to the Hugging Face Hub or local
    storage.

    Args:
        config (Dict[str, Any]):
            A dictionary containing the entire pipeline configuration. Relevant keys:
              - config["pipeline"]["upload_ingest_to_hub"]["run"] (bool):
                  Whether to run this stage.
              - config["pipeline"]["upload_ingest_to_hub"]["source_documents_dir"] (str):
                  Directory containing ingested `.md` files.
              - config["pipeline"]["upload_ingest_to_hub"]["output_dataset_name"] (str, optional):
                  Dataset name for saving or pushing to the Hub. Defaults to
                  config["hf_configuration"]["global_dataset_name"] if unspecified.
              - config["hf_configuration"] (dict):
                  Contains Hugging Face credentials and settings (token, private, etc.).

    Raises:
        ValueError: If required fields in the pipeline configuration are missing.
    """
    stage_name = "upload_ingest_to_hub"
    stage_cfg = config.get("pipeline", {}).get(stage_name, {})

    # Check if this stage is enabled
    if not stage_cfg.get("run", False):
        logger.info(f"Stage '{stage_name}' is disabled. Skipping.")
        return

    # Resolve dataset name and output subset from config or defaults
    output_dataset_name = stage_cfg.get(
        "output_dataset_name",
        config.get("hf_configuration", {}).get("global_dataset_name")
    )
    output_subset = stage_cfg.get("output_subset", stage_name)

    source_dir: str = stage_cfg.get("source_documents_dir")
    if not source_dir:
        raise ValueError(
            f"Missing required config fields in pipeline.{stage_name} "
            f"(needed: 'source_documents_dir')."
        )

    # Logging HF config
    hf_cfg = config.get("hf_configuration", {})
    hf_token: Optional[str] = hf_cfg.get("token")
    hf_private: bool = hf_cfg.get("private", True)

    if not hf_token:
        logger.warning(
            "No Hugging Face token found in 'hf_configuration.token'. "
            "Pushing a private dataset may fail without authentication."
        )

    logger.info("Beginning stage '%s': Uploading ingested Markdown files.", stage_name)
    logger.debug("Source directory for Markdown files: %s", source_dir)
    logger.debug("Hugging Face dataset visibility (private=%s)", hf_private)

    # Gather all markdown files
    markdown_file_paths = glob.glob(os.path.join(source_dir, "*.md"))
    if not markdown_file_paths:
        logger.warning("No .md files found in the specified source directory: %s", source_dir)
        return

    # Convert each file into an IngestedDocument object
    ingested_docs = _create_ingested_documents(markdown_file_paths)
    if not ingested_docs:
        logger.warning("No valid Markdown documents to upload. Exiting stage.")
        return

    # Build a Hugging Face Dataset from these documents
    hf_dataset = _build_hf_dataset(ingested_docs)

    # Finally, save or push the dataset
    save_dataset(
        dataset=hf_dataset,
        step_name=stage_name,
        config=config,
        output_dataset_name=output_dataset_name,
        output_subset=output_subset
    )


def _create_ingested_documents(markdown_file_paths: List[str]) -> List[IngestedDocument]:
    """
    Reads a list of .md file paths and converts them into `IngestedDocument` objects.

    Args:
        markdown_file_paths (List[str]): A list of file paths pointing to
            markdown files.

    Returns:
        List[IngestedDocument]:
            A list of `IngestedDocument` objects, one for each valid .md file.
    """
    ingested_documents: List[IngestedDocument] = []

    for file_path in markdown_file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as file_handle:
                md_content = file_handle.read().strip()

            if not md_content:
                logger.warning("Skipping empty Markdown file: %s", file_path)
                continue

            # Generate a unique ID per file
            unique_doc_id = str(uuid.uuid4())
            file_size_bytes = os.path.getsize(file_path)

            document = IngestedDocument(
                document_id=unique_doc_id,
                document_text=md_content,
                document_filename=os.path.basename(file_path),
                document_metadata={"file_size": file_size_bytes}
            )
            ingested_documents.append(document)

            logger.debug("Successfully created IngestedDocument for file: %s", file_path)

        except Exception as read_error:
            logger.error(
                "Error reading file '%s'. Skipping. Details: %s",
                file_path,
                str(read_error)
            )

    return ingested_documents


def _build_hf_dataset(ingested_documents: List[IngestedDocument]) -> Dataset:
    """
    Converts a list of `IngestedDocument` objects into a Hugging Face `Dataset`.

    Args:
        ingested_documents (List[IngestedDocument]):
            List of documents to be included in the dataset.

    Returns:
        Dataset: A Hugging Face `Dataset` containing the documents' data.
    """
    data_dict = {
        "document_id": [],
        "document_text": [],
        "document_filename": [],
        "document_metadata": [],
    }

    # Populate the dictionary columns with data from each ingested document
    for doc in ingested_documents:
        data_dict["document_id"].append(doc.document_id)
        data_dict["document_text"].append(doc.document_text)
        data_dict["document_filename"].append(doc.document_filename)
        data_dict["document_metadata"].append(doc.document_metadata)

    hf_dataset = Dataset.from_dict(data_dict)
    logger.debug("Created Hugging Face Dataset with %d entries.", len(hf_dataset))

    return hf_dataset
