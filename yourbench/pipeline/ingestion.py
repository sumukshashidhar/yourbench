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
from typing import Dict, Any, Optional

from loguru import logger
from markitdown import MarkItDown

# Attempt to import an OpenAI-like client if available.
# This is only used if advanced LLM-driven conversions are requested.
try:
    from openai import OpenAI
except ImportError:
    # If not installed, fall back gracefully and warn
    logger.warning(
        "Could not import 'openai.OpenAI'. "
        "LLM-based conversion may not be available."
    )
    OpenAI = None

# Add a stage-specific file sink for logging
# Rotation can be tuned as preferred (e.g., daily rotation, size-based, etc.)
os.makedirs("logs", exist_ok=True)
logger.add("logs/ingestion.log", level="DEBUG", rotation="5 MB")


def run(config: Dict[str, Any]) -> None:
    """
    Execute the ingestion stage of the pipeline.

    This function checks whether the ingestion stage is enabled in the pipeline
    configuration. If enabled, it performs the following actions:

    1. Reads all files from the directory specified by `config["pipeline"]["ingestion"]["source_documents_dir"]`.
    2. Converts each file to Markdown using the MarkItDown library.
       Optionally, an LLM can be leveraged for advanced conversions (e.g., image descriptions).
    3. Saves the resulting .md outputs to the directory specified by `config["pipeline"]["ingestion"]["output_dir"]`.

    Args:
        config (Dict[str, Any]): A configuration dictionary with keys:
            - config["pipeline"]["ingestion"]["run"] (bool): Whether to run ingestion.
            - config["pipeline"]["ingestion"]["source_documents_dir"] (str): Directory containing source documents.
            - config["pipeline"]["ingestion"]["output_dir"] (str): Directory where .md files will be saved.
            - config["model_roles"]["ingestion"] (Optional[List[str]]): Model names for LLM ingestion support.
            - config["model_list"] (Optional[List[Dict[str, str]]]): Detailed LLM model configs.

    Returns:
        None

    Logs:
        Writes detailed logs to logs/ingestion.log describing each step taken
        and any errors encountered during file reading or conversion.
    """
    stage_config = config.get("pipeline", {}).get("ingestion", {})
    if not isinstance(stage_config, dict):
        logger.error("Ingestion config is missing or not a dictionary. Aborting ingestion.")
        return

    # Check if ingestion is enabled
    if not stage_config.get("run", False):
        logger.info("Ingestion stage is disabled. No action will be taken.")
        return

    # Extract required directories
    source_dir: Optional[str] = stage_config.get("source_documents_dir")
    output_dir: Optional[str] = stage_config.get("output_dir")

    if not source_dir or not output_dir:
        logger.error("Missing 'source_documents_dir' or 'output_dir' in ingestion config. Cannot proceed.")
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.debug("Prepared output directory: {}", output_dir)

    # Initialize MarkItDown processor (may include LLM if configured)
    markdown_processor = _initialize_markdown_processor(config)

    # Gather all files in the source directory (recursively if desired)
    all_source_files = glob.glob(os.path.join(source_dir, "**"), recursive=True)
    if not all_source_files:
        logger.warning("No files found in source directory: {}", source_dir)
        return

    logger.info(
        "Ingestion stage: Converting files from '{}' to '{}'...",
        source_dir,
        output_dir
    )

    # Process each file in the source directory
    for file_path in all_source_files:
        if os.path.isfile(file_path):
            _convert_document_to_markdown(
                file_path=file_path,
                output_dir=output_dir,
                markdown_processor=markdown_processor
            )

    logger.success(
        "Ingestion stage complete: Processed files from '{}' and saved Markdown to '{}'.",
        source_dir,
        output_dir
    )


def _initialize_markdown_processor(config: Dict[str, Any]) -> MarkItDown:
    """
    Initialize a MarkItDown processor with optional LLM support for advanced conversion.

    This function looks up model details under `config["model_roles"]["ingestion"]`
    and `config["model_list"]` to see if an LLM is defined for ingestion tasks.
    If no suitable model is found or if necessary libraries are missing, a standard
    MarkItDown instance is returned without LLM augmentation.

    Args:
        config (Dict[str, Any]): Global pipeline configuration dictionary.

    Returns:
        MarkItDown: A MarkItDown instance, possibly configured with an LLM client.

    Logs:
        - Warnings if an LLM model is specified but cannot be initialized.
        - Info about which model (if any) is used for ingestion.
    """
    ingestion_role_models = config.get("model_roles", {}).get("ingestion", [])
    model_list = config.get("model_list", [])

    if not ingestion_role_models or not model_list:
        logger.debug("No LLM ingestion config found. Using default MarkItDown processor.")
        return MarkItDown()

    # Attempt to match the first model in model_list that appears in ingestion_role_models
    matched_model_info = next(
        (m for m in model_list if m["model_name"] in ingestion_role_models),
        None
    )

    if not matched_model_info:
        logger.debug(
            "No matching LLM model found for roles: {}. Using default MarkItDown.",
            ingestion_role_models
        )
        return MarkItDown()

    # If the openai library is not available, fallback
    if OpenAI is None:
        logger.warning(
            "OpenAI library is not available; cannot initialize LLM for ingestion. Using default MarkItDown."
        )
        return MarkItDown()

    # Extract relevant info from the matched model config
    request_style = matched_model_info.get("request_style", "")
    base_url = matched_model_info.get("base_url", "")
    api_key = matched_model_info.get("api_key", "")
    model_name = matched_model_info.get("model_name", "unknown_model")

    # Expand environment variables in the api_key, if present
    api_key = os.path.expandvars(api_key) if api_key else ""

    logger.info(
        "Initializing MarkItDown with LLM support: request_style='{}', model='{}'.",
        request_style,
        model_name
    )

    # Construct the LLM client (placeholder usage, adjust to real client as needed)
    llm_client = OpenAI(api_key=api_key, base_url=base_url)  # Example usage
    return MarkItDown(llm_client=llm_client, llm_model=model_name)


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

        logger.info(
            "Successfully converted '%s' -> '%s'.",
            file_path,
            output_file
        )
    except Exception as exc:
        logger.error(
            "Failed to convert '%s'. Error details: %s",
            file_path,
            exc
        )