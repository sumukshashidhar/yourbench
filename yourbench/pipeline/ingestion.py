"""
Ingestion Module

This module handles the "ingestion" stage of the Yourbench pipeline. It reads raw source
documents from a specified directory, converts each file to Markdown using MarkItDown,
and saves the output Markdown files to an output directory. If an LLM is configured for
image descriptions or more advanced conversions, it is integrated via MarkItDown’s
`llm_client` and `llm_model`.

Usage:
    # Example usage within the Yourbench pipeline
    from ingestion import run

    ingestion_config = {
        "source_documents_dir": "data/example/raw",
        "output_dir": "data/example/ingested",
        "run": True
    }
    run({"pipeline": {"ingestion": ingestion_config}})
"""

import os
import glob
import shutil
from typing import Dict, Any, Optional

from loguru import logger
from markitdown import MarkItDown

# If you have a custom OpenAI/OpenRouter Python client, import here.
# For demonstration, we use "openai.OpenAI" as a placeholder.
# If you have a different library or custom code, adjust accordingly.
try:
    from openai import OpenAI
except ImportError:
    logger.warning("Could not import 'openai.OpenAI'. Please ensure it's installed if you need LLM features.")
    OpenAI = None  # Fallback if library isn't installed


def run(config: Dict[str, Any]) -> None:
    """
    Execute the ingestion stage of the pipeline.

    This function checks the pipeline configuration for the ingestion stage. If the user
    has enabled the ingestion stage (run == True), it proceeds to:
      1. Read the source_documents_dir for all files.
      2. For each file, convert it to Markdown using MarkItDown.
      3. Save the output (.md) to the output_dir.

    If an LLM is specified in the configuration (through model_roles and model_list),
    MarkItDown can use this LLM for advanced conversions (e.g., generating image captions).

    Parameters:
        config (Dict[str, Any]): A dictionary containing overall pipeline configuration.
                                 Expected structure:
                                 {
                                   "pipeline": {
                                     "ingestion": {
                                       "source_documents_dir": str,
                                       "output_dir": str,
                                       "run": bool
                                     }
                                   },
                                   "model_roles": {
                                     "ingestion": [list_of_model_keys_for_ingestion]
                                   },
                                   "model_list": [
                                     {
                                       "model_name": str,
                                       "provider": str,
                                       "base_url": str,
                                       "api_key": str
                                     },
                                     ...
                                   ]
                                 }

    Returns:
        None
    """
    # === Validate and extract ingestion configuration ===
    ingestion_cfg = config.get("pipeline", {}).get("ingestion", {})
    if not ingestion_cfg:
        logger.warning("No ingestion configuration found. Skipping ingestion stage.")
        return

    if not ingestion_cfg.get("run", False):
        logger.info("Ingestion stage disabled. Skipping.")
        return

    source_dir = ingestion_cfg.get("source_documents_dir")
    output_dir = ingestion_cfg.get("output_dir")
    if not source_dir or not output_dir:
        logger.error("source_documents_dir or output_dir not specified. Cannot proceed.")
        return

    # === Prepare output directory ===
    os.makedirs(output_dir, exist_ok=True)
    logger.debug("Ensured output directory exists: {}", output_dir)

    # === (Optional) Resolve a model for advanced image descriptions, etc. ===
    md = _initialize_markitdown_with_llm(config)

    # === Convert each file in the source directory ===
    file_paths = glob.glob(os.path.join(source_dir, "**"), recursive=True)
    if not file_paths:
        logger.warning("No files found in source directory: {}", source_dir)
        return

    logger.info(
        "Starting ingestion: converting files from '{}' to '{}'.",
        source_dir,
        output_dir
    )

    for fp in file_paths:
        if os.path.isfile(fp):
            _convert_file(fp, output_dir, md)

    logger.success("Ingestion complete. Processed files from '{}' to '{}'.", source_dir, output_dir)


def _initialize_markitdown_with_llm(config: Dict[str, Any]) -> MarkItDown:
    """
    Optionally initialize MarkItDown with LLM client settings if configured.

    If the ingestion stage has associated model roles that specify an LLM (e.g., for
    image descriptions), this function attempts to read those settings from the config
    and initialize MarkItDown accordingly. If no relevant model is found or if the
    openai.OpenAI library isn't installed, returns a basic MarkItDown instance.

    Parameters:
        config (Dict[str, Any]): Overall pipeline configuration.

    Returns:
        MarkItDown: An instance of MarkItDown, possibly with an LLM client attached.
    """

    # Try retrieving the model role for ingestion
    ingestion_roles = config.get("model_roles", {}).get("ingestion", [])
    model_list = config.get("model_list", [])

    # If there's no model specified for ingestion, just return a basic MarkItDown instance
    if not ingestion_roles or not model_list:
        logger.debug("No LLM configuration found for ingestion. Using default MarkItDown.")
        return MarkItDown(enable_plugins=False)

    # For simplicity, pick the first model in ingestion_roles that matches anything in model_list
    # (Feel free to adapt for more complex logic if needed)
    matched_model_config: Optional[Dict[str, Any]] = None
    for role_key in ingestion_roles:
        for model_conf in model_list:
            if model_conf.get("model_name") == role_key:
                matched_model_config = model_conf
                break
        if matched_model_config:
            break

    if not matched_model_config:
        logger.debug("No matching LLM config found for roles: {}. Using default MarkItDown.", ingestion_roles)
        return MarkItDown(enable_plugins=False)

    # Attempt to initialize the LLM client
    if OpenAI is None:
        logger.warning(
            "OpenAI client library not found. Unable to initialize LLM. Using default MarkItDown."
        )
        return MarkItDown(enable_plugins=False)

    # Extract relevant info
    provider = matched_model_config.get("provider")
    base_url = matched_model_config.get("base_url")
    api_key = matched_model_config.get("api_key")
    model_name = matched_model_config.get("model_name")  # e.g. "gemini_flash" or something similar

    # Expand any environment variables in the api_key string
    if api_key:
        api_key = os.path.expandvars(api_key)

    # Log the chosen model
    logger.info(
        "Initializing MarkItDown with LLM support: provider='{}', model='{}'.",
        provider,
        model_name
    )

    # Construct the LLM client (this is an example, adjust to your actual client interface)
    llm_client = OpenAI(api_key=api_key, base_url=base_url)
    return MarkItDown(llm_client=llm_client, llm_model=model_name)


def _convert_file(file_path: str, output_dir: str, md: MarkItDown) -> None:
    """
    Convert a single file to Markdown and save the output.

    Parameters:
        file_path (str): Path to the source file to be converted.
        output_dir (str): Directory where the resulting .md file should be saved.
        md (MarkItDown): A configured MarkItDown instance.
    """
    logger.debug("Converting file: {}", file_path)
    try:
        # Convert the file to Markdown
        result = md.convert(file_path)

        # Construct an output filename with .md extension
        base_name = os.path.basename(file_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_dir, f"{file_name_no_ext}.md")

        # Save the converted text content
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(result.text_content)

        logger.info("Successfully converted '{}' -> '{}'.", file_path, output_file)
    except Exception as exc:
        logger.error("Failed to convert '{}'. Error: {}", file_path, exc)


if __name__ == "__main__":
    # This is just a placeholder if you run this module directly.
    # Typically, you'd import and call `run(config)` from another part of your application.
    example_config = {
        "pipeline": {
            "ingestion": {
                "source_documents_dir": "data/example/raw",
                "output_dir": "data/example/ingested",
                "run": True
            }
        },
        "model_roles": {
            "ingestion": ["gemini_flash"]
        },
        "model_list": [
            {
                "model_name": "gemini_flash",
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "$OPENROUTER_API_KEY"
            }
        ]
    }
    run(example_config)
