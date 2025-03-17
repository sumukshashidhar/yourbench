import os
import glob
from typing import Dict, Any, List
from dataclasses import dataclass, field
import uuid

from loguru import logger
from datasets import Dataset
from yourbench.utils.loading_engine import load_config
from yourbench.utils.dataset_engine import save_dataset

@dataclass
class IngestedDocument:
    """Represents an ingested document."""
    document_id: str
    document_text: str
    document_filename: str
    document_metadata: Dict[str, Any] = field(default_factory=dict)

def run(config: Dict[str, Any]) -> None:
    """
    Upload ingested markdown files to Hugging Face Hub as a dataset.

    Expected structure:
      quant_test
      ├── ingested
      │   ├── train
      │   └── test
    """
    stage_name = "upload_ingest_to_hub"
    stage_cfg = config.get("pipeline", {}).get(stage_name, {})
    output_dataset_name = stage_cfg.get("output_dataset_name", config.get("hf_configuration", {}).get("global_dataset_name"))
    output_subset = stage_cfg.get("output_subset", stage_name)

    if not stage_cfg.get("run", False):
        logger.info(f"Stage '{stage_name}' is disabled. Skipping.")
        return

    source_dir: str = stage_cfg.get("source_documents_dir")
    # hub_dataset_name: str = stage_cfg.get("hub_dataset_name")
    # local_dataset_path: Optional[str] = stage_cfg.get("local_dataset_path")

    if not source_dir:
        raise ValueError(
            f"Missing required config fields in pipeline.{stage_name} "
            f"(needed: 'source_documents_dir')."
        )

    # Global Hugging Face configuration
    hf_cfg = config.get("hf_configuration", {})
    hf_token: Optional[str] = hf_cfg.get("token")
    hf_private: bool = hf_cfg.get("private", True)

    if not hf_token:
        logger.warning(
            "No Hugging Face token found in 'hf_configuration.token'. "
            "Attempting push without auth. If your dataset is private, this will fail."
        )

    logger.info("Starting '{}' stage: uploading ingested files to HF Hub.", stage_name)
    logger.debug(f"Source directory: {source_dir}")
    logger.debug(f"HF dataset visibility set to private={hf_private}")

    # === Collect .md files and create IngestedDocument objects ===
    md_file_paths = glob.glob(os.path.join(source_dir, "*.md"))
    if not md_file_paths:
        logger.warning(f"No .md files found in source directory: {source_dir}")
        return

    ingested_documents = _load_md_files_as_dataclasses(md_file_paths)

    if not ingested_documents:
        logger.warning("No valid markdown files to upload. Exiting stage.")
        return

    dataset = _create_hf_dataset_from_ingested_documents(ingested_documents)

    save_dataset(
        dataset = dataset,
        step_name = stage_name,
        config = config,
        output_dataset_name = output_dataset_name,
        output_subset = output_subset
    )
    return


def _load_md_files_as_dataclasses(md_file_paths: List[str]) -> List[IngestedDocument]:
    """
    Given a list of .md file paths, read file content and create IngestedDocument objects.
    """
    ingested_docs: List[IngestedDocument] = []
    for file_path in md_file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                logger.warning("Skipping empty markdown file: {}", file_path)
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
            logger.debug("Loaded IngestedDocument for file: {}", file_path)

        except Exception as e:
            logger.error(
                "Error reading file '{}'. Skipping. Error details: {}",
                file_path,
                str(e)
            )

    return ingested_docs


def _create_hf_dataset_from_ingested_documents(
    ingested_docs: List[IngestedDocument]   
) -> Dataset:
    """
    Convert a list of IngestedDocument objects into a Hugging Face Dataset.
    """
    data_dict = {
        "document_id": [],
        "document_text": [],
        "document_filename": [],
        "document_metadata": [],
    }

    for doc in ingested_docs:
        data_dict["document_id"].append(doc.document_id)
        data_dict["document_text"].append(doc.document_text)
        data_dict["document_filename"].append(doc.document_filename)
        data_dict["document_metadata"].append(doc.document_metadata)

    dataset = Dataset.from_dict(data_dict)
    logger.debug("Created Hugging Face Dataset with {} entries.", len(dataset))
    return dataset
