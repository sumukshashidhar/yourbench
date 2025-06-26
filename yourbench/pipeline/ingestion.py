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
      "llm_ingestion": bool,       # Whether to use LLM for PDF ingestion
      "pdf_batch_size": int,       # Number of PDF pages to process in parallel
      "pdf_dpi": int,              # DPI for PDF to image conversion
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

import io
import os
import glob
import base64
from typing import Any, Optional
from pathlib import Path
from dataclasses import field, dataclass

import trafilatura
from PIL import Image
from loguru import logger
from pdf2image import convert_from_path
from markitdown import MarkItDown

from huggingface_hub import InferenceClient
from yourbench.utils.inference.inference_core import Model as ModelConfig
from yourbench.utils.inference.inference_core import InferenceCall, run_inference


@dataclass
class IngestionConfig:
    """Configuration for the ingestion stage of the pipeline."""

    run: bool = True
    source_documents_dir: Optional[str] = None
    output_dir: Optional[str] = None
    llm_ingestion: bool = False  # Toggle for LLM-based PDF ingestion
    pdf_batch_size: int = 5
    pdf_dpi: int = 300


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
        llm_ingestion=stage_config.get("llm_ingestion", False),
        pdf_batch_size=stage_config.get("pdf_batch_size", 5),
        pdf_dpi=stage_config.get("pdf_dpi", 300),
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


def _pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[Image.Image]:
    """Convert PDF to list of PIL images."""
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        logger.info(f"Converted {pdf_path.name} to {len(images)} images")
        return images
    except Exception as e:
        logger.error(f"Failed to convert PDF {pdf_path}: {e}")
        return []


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string."""

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _build_pdf_inference_calls(pdf_path: Path, images: list[Image.Image]) -> tuple[list[InferenceCall], list[int]]:
    """Build inference calls for PDF pages."""
    calls = []
    page_numbers = []

    for page_num, image in enumerate(images, start=1):
        image_b64 = _image_to_base64(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Convert this document page to clean, well-formatted Markdown. "
                            "Preserve all text, structure, tables, and important formatting. "
                            "Do not add any commentary or metadata - just the content in Markdown."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ]

        calls.append(InferenceCall(messages=messages, tags=["pdf_ingestion", f"page_{page_num}", pdf_path.name]))
        page_numbers.append(page_num)

    logger.info(f"Created {len(calls)} inference calls for PDF {pdf_path.name}")
    return calls, page_numbers


def _process_pdf_with_llm(pdf_path: Path, config: dict[str, Any], ingestion_config: IngestionConfig) -> str:
    """Process entire PDF through LLM using the inference engine."""
    logger.info(f"Processing PDF with LLM: {pdf_path.name}")

    # Convert PDF to images
    images = _pdf_to_images(pdf_path, ingestion_config.pdf_dpi)
    if not images:
        return ""

    # Build inference calls
    inference_calls, page_numbers = _build_pdf_inference_calls(pdf_path, images)

    # Run inference using the engine
    responses = run_inference(config=config, step_name="ingestion", inference_calls=inference_calls)

    # Process responses
    if not responses:
        logger.error(f"No responses received for PDF {pdf_path.name}")
        return ""

    # Get the first (and likely only) model's responses
    model_name = list(responses.keys())[0]
    page_contents = responses[model_name]

    # Sort and combine pages
    pages_with_numbers = list(zip(page_numbers, page_contents))
    pages_with_numbers.sort(key=lambda x: x[0])

    # Concatenate with page breaks
    markdown_pages = [content for _, content in pages_with_numbers if content]
    full_markdown = "\n\n---\n\n".join(markdown_pages)

    logger.success(f"Successfully processed {len(images)} pages from {pdf_path.name}")
    return full_markdown


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
            logger.info(f"No matching LLM model found for roles: {model_roles.ingestion}. Using default MarkItDown.")
            return MarkItDown()

        logger.info(f"Initializing MarkItDown with LLM support: model='{matched_model.model_name}'.")

        # Construct the InferenceClient client (as OpenAI replacement)
        llm_client = InferenceClient(
            base_url=matched_model.base_url,
            api_key=matched_model.api_key,
            provider=matched_model.provider,
        )

        return MarkItDown(llm_client=llm_client, llm_model=matched_model.model_name)
    except Exception as exc:
        logger.warning(f"Failed to initialize MarkItDown with LLM support: {str(exc)}")
        return MarkItDown()


def _extract_markdown_from_html(file_path: str) -> str | None:
    """Attempts to extract markdown content from an HTML file using Trafilatura."""
    logger.debug(f"Attempting to extract Markdown from HTML file: {file_path} using Trafilatura.")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # output_format='markdown' is key for direct Markdown conversion
        extracted_markdown = trafilatura.extract(
            html_content,
            output_format="markdown",
            include_comments=False,  # Do not include HTML comments
            include_tables=True,  # Try to include table data
        )

        if extracted_markdown:
            logger.info(f"Successfully extracted Markdown from '{file_path}' using Trafilatura.")
            return extracted_markdown

        logger.warning(f"Trafilatura returned no content for HTML file '{file_path}'.")
        return None
    except Exception as e:
        logger.error(f"Error using Trafilatura for HTML file '{file_path}': {e}. Skipping Trafilatura for this file.")
        return None


def _get_markdown_content(
    file_path: str, 
    markdown_processor: MarkItDown,
    config: dict[str, Any] = None,
    ingestion_config: IngestionConfig = None
) -> str | None:
    """
    Extract or convert file content to Markdown based on file type.

    Args:
        file_path (str): The path to the source document.
        markdown_processor (MarkItDown): Configured MarkItDown instance for conversions.
        config (dict[str, Any]): Full pipeline config (needed for LLM PDF processing).
        ingestion_config (IngestionConfig): Ingestion configuration.

    Returns:
        str | None: The Markdown content, or None if conversion failed.

    Logs:
        - Info about the processing method used for each file type.
        - Warnings for fallback scenarios or failed conversions.
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == ".md":
        # For existing Markdown files, just read the content, ensuring UTF-8
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"File '{file_path}' is already Markdown. Content read directly.")
        return content

    elif file_ext in [".html", ".htm"]:
        logger.info(f"Processing HTML file: {file_path} with Trafilatura.")
        content = _extract_markdown_from_html(file_path)
        if content is None:  # Fallback to MarkItDown if Trafilatura failed or returned nothing
            logger.warning(
                f"Trafilatura processing failed or yielded no content for HTML '{file_path}'. Falling back to MarkItDown for this file."
            )
            content = markdown_processor.convert(file_path).text_content
        return content

    elif file_ext == ".pdf":
        # Check if LLM ingestion is enabled and configured
        if ingestion_config and ingestion_config.llm_ingestion and config:
            logger.info(f"Processing PDF '{file_path}' with LLM ingestion.")
            return _process_pdf_with_llm(Path(file_path), config, ingestion_config)
        else:
            logger.info(f"Processing PDF '{file_path}' with MarkItDown (LLM ingestion disabled).")
            return markdown_processor.convert(file_path).text_content

    else:  # For other file types, use the MarkItDown processor
        logger.info(f"Converting non-HTML/Markdown file '{file_path}' using MarkItDown.")
        return markdown_processor.convert(file_path).text_content


def _convert_document_to_markdown(
    file_path: str, 
    output_dir: str, 
    markdown_processor: MarkItDown,
    config: dict[str, Any] = None,
    ingestion_config: IngestionConfig = None
) -> None:
    """
    Convert a single source file into Markdown and save the result.

    Args:
        file_path (str): The path to the source document.
        output_dir (str): Directory where the converted .md file will be written.
        markdown_processor (MarkItDown): Configured MarkItDown instance for conversions.
        config (dict[str, Any]): Full pipeline config (needed for LLM PDF processing).
        ingestion_config (IngestionConfig): Ingestion configuration.

    Returns:
        None

    Logs:
        - Debug info about the file being processed.
        - Warning if conversion fails or the file is empty.
    """
    logger.debug(f"Converting file: {file_path}")
    try:
        content = _get_markdown_content(file_path, markdown_processor, config, ingestion_config)

        if content is None:
            logger.warning(f"No content could be generated for file '{file_path}' after processing. Skipping output.")
            return

        # Construct an output filename with .md extension
        base_name = os.path.basename(file_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_dir, f"{file_name_no_ext}.md")

        # Write the converted Markdown to disk
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(content)

        logger.info(f"Successfully processed '{file_path}' and saved as '{output_file}'.")
    except Exception as exc:
        logger.error(f"Failed to convert '{file_path}'. Error details: {exc}")


def _validate_llm_ingestion_config(config: dict[str, Any], ingestion_config: IngestionConfig) -> bool:
    """
    Validate that LLM ingestion is properly configured when enabled.
    
    Returns:
        bool: True if configuration is valid, False otherwise.
    """
    if not ingestion_config.llm_ingestion:
        return True  # No validation needed if disabled
    
    model_roles = _extract_model_roles(config)
    model_list = _extract_model_list(config)
    
    if not model_roles.ingestion:
        logger.error("LLM ingestion is enabled but no models are assigned to 'ingestion' role in model_roles.")
        return False
    
    if not model_list:
        logger.error("LLM ingestion is enabled but no models are defined in model_list.")
        return False
    
    # Check if at least one ingestion model exists in model_list
    matched_models = [m for m in model_list if m.model_name in model_roles.ingestion]
    if not matched_models:
        logger.error(f"LLM ingestion is enabled but none of the models in model_roles.ingestion {model_roles.ingestion} are found in model_list.")
        return False
    
    logger.info(f"LLM ingestion validated. Found {len(matched_models)} model(s) for ingestion.")
    return True


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
            - config["pipeline"]["ingestion"]["llm_ingestion"] (bool): Whether to use LLM for PDF ingestion.
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
    
    # Validate LLM configuration if LLM ingestion is enabled
    if not _validate_llm_ingestion_config(config, ingestion_config):
        logger.error("LLM ingestion validation failed. Please check your configuration.")
        return

    # Ensure the output directory exists
    os.makedirs(ingestion_config.output_dir, exist_ok=True)
    logger.debug(f"Prepared output directory: {ingestion_config.output_dir}")

    # Initialize MarkItDown processor (may include LLM if configured)
    markdown_processor = _initialize_markdown_processor(config)

    # Gather all files in the source directory (recursively if desired)
    all_source_files = glob.glob(os.path.join(ingestion_config.source_documents_dir, "**"), recursive=True)
    if not all_source_files:
        logger.warning(f"No files found in source directory: {ingestion_config.source_documents_dir}")
        return

    logger.info(
        f"Ingestion stage: Converting files from '{ingestion_config.source_documents_dir}' to '{ingestion_config.output_dir}'..."
    )
    
    if ingestion_config.llm_ingestion:
        logger.info("LLM ingestion mode is ENABLED for PDF files.")
    else:
        logger.info("LLM ingestion mode is DISABLED. PDFs will be processed with MarkItDown.")

    # Process each file in the source directory
    for file_path in all_source_files:
        if os.path.isfile(file_path):
            _convert_document_to_markdown(
                file_path=file_path,
                output_dir=ingestion_config.output_dir,
                markdown_processor=markdown_processor,
                config=config,
                ingestion_config=ingestion_config,
            )

    logger.success(
        f"Ingestion stage complete: Processed files from '{ingestion_config.source_documents_dir}' and saved Markdown to '{ingestion_config.output_dir}'."
    )