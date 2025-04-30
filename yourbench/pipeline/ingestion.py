# ingestion.py

"""
Author: @sumukshashidhar

This module implements the "ingestion" stage of the YourBench pipeline.

Purpose:
    The ingestion stage reads source documents from a user-specified directory,
    converts each document into markdown (optionally assisted by an LLM), and
    saves the converted outputs in the specified output directory. This normalized
    markdown output sets the foundation for subsequent pipeline steps.

Usage:
    from yourbench.pipeline import ingestion
    ingestion.run(config)

Configuration Requirements (in `config["pipeline"]["ingestion"]`):
    {
      "run": bool,  # Whether to enable the ingestion stage
      "source_documents_dir": str,  # Directory containing raw source documents
      "output_dir": str,           # Directory where converted .md files are saved
    }

    Additionally, LLM details can be defined in:
    config["model_roles"]["ingestion"] = [list_of_model_names_for_ingestion]
    config["model_list"] = [
      {
        "model_name": str,
        "request_style": str,
        "base_url": str,
        "api_key": str,
        ...
      },
      ...
    ]

Stage-Specific Logging:
    All major ingestion activity is logged to "logs/ingestion.log".
"""

import os
import glob
from typing import Any, Optional
from dataclasses import field, dataclass

from loguru import logger
from markitdown import MarkItDown

from huggingface_hub import InferenceClient
from yourbench.utils.inference_engine import Model as ModelConfig


@dataclass
class IngestionConfig:
    """Configuration for the ingestion stage of the pipeline."""

    run: bool = True
    source_documents_dir: Optional[str] = None
    output_dir: Optional[str] = None


@dataclass
class ModelRoles:
    """Configuration for model roles in the pipeline."""

    ingestion: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Main configuration for the pipeline."""

    pipeline: dict[str, Any] = field(default_factory=dict)
    model_roles: ModelRoles = field(default_factory=ModelRoles)
    model_list: list[ModelConfig] = field(default_factory=list)


def _extract_ingestion_config(config: dict[str, Any]) -> IngestionConfig:
    """
    Extract ingestion configuration from the main config dictionary.

    Args:
        config (dict[str, Any]): The complete configuration dictionary.

    Returns:
        IngestionConfig: A typed configuration object for ingestion.
    """
    if not isinstance(config.get("pipeline", {}).get("ingestion", {}), dict):
        return IngestionConfig()

    stage_config = config.get("pipeline", {}).get("ingestion", {})
    return IngestionConfig(
        run=stage_config.get("run", True),
        source_documents_dir=stage_config.get("source_documents_dir"),
        output_dir=stage_config.get("output_dir"),
    )


def _extract_model_roles(config: dict[str, Any]) -> ModelRoles:
    """
    Extract model roles configuration from the main config dictionary.

    Args:
        config (dict[str, Any]): The complete configuration dictionary.

    Returns:
        ModelRoles: A typed configuration object for model roles.
    """
    model_roles_dict = config.get("model_roles", {})
    return ModelRoles(ingestion=model_roles_dict.get("ingestion", []))


def _extract_model_list(config: dict[str, Any]) -> list[ModelConfig]:
    """
    Extract model list configuration from the main config dictionary.

    Args:
        config (dict[str, Any]): The complete configuration dictionary.

    Returns:
        list[ModelConfig]: A list of typed model configurations.
    """
    model_list_dicts = config.get("model_list", [])
    result = []

    for model_dict in model_list_dicts:
        model_config = ModelConfig(
            model_name=model_dict.get("model_name"),
            base_url=model_dict.get("base_url"),
            api_key=model_dict.get("api_key"),
            provider=model_dict.get("provider"),
        )
        result.append(model_config)

    return result


def run(config: dict[str, Any]) -> None:
    """
    Execute the ingestion stage of the pipeline.

    This function checks whether the ingestion stage is enabled in the pipeline
    configuration. If enabled, it performs the following actions:

    1. Reads all files from the directory specified by `config["pipeline"]["ingestion"]["source_documents_dir"]`.
    2. Converts each file to Markdown using the MarkItDown library.
       Optionally, an LLM can be leveraged for advanced conversions (e.g., image descriptions).
    3. Saves the resulting .md outputs to the directory specified by `config["pipeline"]["ingestion"]["output_dir"]`.

    Args:
        config (dict[str, Any]): A configuration dictionary with keys:
            - config["pipeline"]["ingestion"]["run"] (bool): Whether to run ingestion.
            - config["pipeline"]["ingestion"]["source_documents_dir"] (str): Directory containing source documents.
            - config["pipeline"]["ingestion"]["output_dir"] (str): Directory where .md files will be saved.
            - config["model_roles"]["ingestion"] (Optional[list[str]]): Model names for LLM ingestion support.
            - config["model_list"] (Optional[list[dict[str, str]]]): Detailed LLM model configs.

    Returns:
        None

    Logs:
        Writes detailed logs to logs/ingestion.log describing each step taken
        and any errors encountered during file reading or conversion.
    """
    # Extract typed configurations from the dictionary
    ingestion_config = _extract_ingestion_config(config)

    # Check if ingestion is enabled
    if not ingestion_config.run:
        logger.info("Ingestion stage is disabled. No action will be taken.")
        return

    # Check required directories
    if not ingestion_config.source_documents_dir or not ingestion_config.output_dir:
        logger.error("Missing 'source_documents_dir' or 'output_dir' in ingestion config. Cannot proceed.")
        return

    # Ensure the output directory exists
    os.makedirs(ingestion_config.output_dir, exist_ok=True)
    logger.debug("Prepared output directory: {}", ingestion_config.output_dir)

    # Initialize MarkItDown processor (may include LLM if configured)
    markdown_processor = _initialize_markdown_processor(config)

    # Gather all files in the source directory (recursively if desired)
    all_source_files = glob.glob(os.path.join(ingestion_config.source_documents_dir, "**"), recursive=True)
    if not all_source_files:
        logger.warning(
            "No files found in source directory: {}",
            ingestion_config.source_documents_dir,
        )
        return

    logger.info(
        "Ingestion stage: Converting files from '{}' to '{}'...",
        ingestion_config.source_documents_dir,
        ingestion_config.output_dir,
    )

    # Process each file in the source directory
    for file_path in all_source_files:
        if os.path.isfile(file_path):
            _convert_document_to_markdown(
                file_path=file_path,
                output_dir=ingestion_config.output_dir,
                markdown_processor=markdown_processor,
            )

    logger.success(
        "Ingestion stage complete: Processed files from '{}' and saved Markdown to '{}'.",
        ingestion_config.source_documents_dir,
        ingestion_config.output_dir,
    )


def _initialize_markdown_processor(config: dict[str, Any]) -> MarkItDown:
    """
    Initialize a MarkItDown processor with optional LLM support for advanced conversion.

    This function looks up model details under `config["model_roles"]["ingestion"]`
    and `config["model_list"]` to see if an LLM is defined for ingestion tasks.
    If no suitable model is found or if necessary libraries are missing, a standard
    MarkItDown instance is returned without LLM augmentation.

    Args:
        config (dict[str, Any]): Global pipeline configuration dictionary.

    Returns:
        MarkItDown: A MarkItDown instance, possibly configured with an LLM client.

    Logs:
        - Warnings if an LLM model is specified but cannot be initialized.
        - Info about which model (if any) is used for ingestion.
    """
    try:
        # Extract typed configurations from the dictionary
        model_roles = _extract_model_roles(config)
        model_list = _extract_model_list(config)

        if not model_roles.ingestion or not model_list:
            logger.info("No LLM ingestion config found. Using default MarkItDown processor.")
            return MarkItDown()

        # Attempt to match the first model in model_list that appears in model_roles.ingestion
        matched_model = next((m for m in model_list if m.model_name in model_roles.ingestion), None)

        if not matched_model:
            logger.info(
                "No matching LLM model found for roles: {}. Using default MarkItDown.",
                model_roles.ingestion,
            )
            return MarkItDown()

        logger.info(
            "Initializing MarkItDown with LLM support: model='{}'.",
            matched_model.model_name,
        )

        # Construct the InferenceClient client (as OpenAI replacement)
        llm_client = InferenceClient(
            base_url=matched_model.base_url,
            api_key=matched_model.api_key,
            provider=matched_model.provider,
        )

        return MarkItDown(llm_client=llm_client, llm_model=matched_model.model_name)
    except Exception as exc:
        logger.warning("Failed to initialize MarkItDown with LLM support: {}", str(exc))
        return MarkItDown()


def _convert_document_to_markdown(file_path: str, output_dir: str, markdown_processor: MarkItDown) -> None:
    """
    Convert a single source file into Markdown using MarkItDown and save the result.

    Args:
        file_path (str): The path to the source document.
        output_dir (str): Directory where the converted .md file will be written.
        markdown_processor (MarkItDown): Configured MarkItDown instance to handle the conversion.

    Returns:
        None

    Logs:
        - Debug info about the file being processed.
        - Warning if conversion fails or the file is empty.
    """
    logger.debug("Converting file: {}", file_path)
    try:
        # Perform the file-to-markdown conversion
        conversion_result = markdown_processor.convert(file_path)

        # Construct an output filename with .md extension
        base_name = os.path.basename(file_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_dir, f"{file_name_no_ext}.md")

        # Write the converted Markdown to disk
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(conversion_result.text_content)

        logger.info(f"Successfully converted '{file_path}' -> '{output_file}'.")
    except Exception as exc:
        logger.error(f"Failed to convert '{file_path}'. Error details: {exc}")
