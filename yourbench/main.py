#!/usr/bin/env python3
"""YourBench CLI - Dynamic Evaluation Set Generation with Large Language Models."""

from __future__ import annotations
import os
import sys
import json
import time
from typing import Optional
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger
from randomname import get_name as get_random_name


# Track startup time
startup_time = time.perf_counter()

# Configure logging early
logger.remove()
logger.add(sys.stderr, level=os.getenv("YOURBENCH_LOG_LEVEL", "INFO"))

# Early startup logging
logger.debug("YourBench starting up...")
logger.debug("Loading core modules...")

# Lazy imports - only import when needed
run_pipeline = None


def _lazy_import_pipeline():
    global run_pipeline
    if run_pipeline is None:
        logger.debug("Loading pipeline components...")
        from yourbench.pipeline.handler import run_pipeline as _run_pipeline

        run_pipeline = _run_pipeline
    return run_pipeline


logger.debug("Loading environment variables...")
load_dotenv()

app = typer.Typer(
    name="yourbench",
    help="YourBench - Dynamic Evaluation Set Generation with Large Language Models.",
    pretty_exceptions_show_locals=False,
    add_completion=False,
)

# Log startup completion
logger.debug(f"YourBench loaded in {time.perf_counter() - startup_time:.2f}s")


def _parse_extra_parameters(raw_value: Optional[str]) -> dict:
    """Parse CLI-supplied JSON for model extra parameters."""
    if not raw_value:
        return {}

    candidate = raw_value.strip()
    if not candidate:
        return {}

    # Allow passing a file path for convenience
    possible_path = Path(candidate)
    if possible_path.exists() and possible_path.is_file():
        try:
            candidate = possible_path.read_text(encoding="utf-8")
        except Exception as exc:
            raise typer.BadParameter(f"Failed to read extra parameter file '{candidate}': {exc}")

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON for model extra parameters: {exc}") from exc

    if not isinstance(parsed, dict):
        raise typer.BadParameter("Model extra parameters JSON must decode to a JSON object")

    return parsed


def run_yourbench(
    config_or_docs: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_extra_parameters: Optional[str] = None,
    provider: Optional[str] = None,
    single_shot_questions: Optional[bool] = None,
    multi_hop_questions: bool = False,
    cross_doc_questions: bool = False,
    additional_instructions: Optional[str] = None,
    push_to_hub: Optional[str] = None,
    debug: bool = False,
    output_dir: Optional[Path] = None,
    # Additional pipeline configuration options
    question_mode: Optional[str] = None,
    max_tokens: Optional[int] = None,
    token_overlap: Optional[int] = None,
    l_max_tokens: Optional[int] = None,
    h_min: Optional[int] = None,
    h_max: Optional[int] = None,
    pdf_dpi: Optional[int] = None,
    llm_ingestion: bool = False,
    question_rewriting: bool = False,
    private_dataset: bool = True,
    local_saving: bool = True,
    export_jsonl: bool = True,
    jsonl_export_dir: Optional[Path] = None,
    max_concurrent_requests: Optional[int] = None,
    encoding_name: Optional[str] = None,
    bill_to: Optional[str] = None,
) -> None:
    """Core function to run YourBench pipeline."""
    # Setup debug logging if requested
    if debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # Convert to Path
    config_or_docs_path = Path(config_or_docs)

    # Check if input is a YAML config file
    if config_or_docs_path.suffix in [".yaml", ".yml"] and config_or_docs_path.exists():
        logger.info(f"Running with config: {config_or_docs_path}")
        if model_extra_parameters:
            logger.warning("Ignoring --model-extra-parameters because an explicit configuration file was provided")

        # Run pipeline with existing config
        pipeline_func = _lazy_import_pipeline()
        try:
            pipeline_func(config_file_path=str(config_or_docs_path), debug=debug)
        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            raise typer.Exit(1)
    else:
        # Generate config dynamically for document path
        if not config_or_docs_path.exists():
            logger.error(f"Path does not exist: {config_or_docs}")
            raise typer.Exit(1)

        # Import configuration classes
        from yourbench.utils.configuration_engine import (
            ModelConfig,
            ChunkingConfig,
            PipelineConfig,
            IngestionConfig,
            LightevalConfig,
            YourbenchConfig,
            HuggingFaceConfig,
            SummarizationConfig,
            QuestionRewritingConfig,
            MultiHopQuestionGenerationConfig,
            SingleShotQuestionGenerationConfig,
            CrossDocumentQuestionGenerationConfig,
        )

        # Create configuration programmatically
        dataset_name = push_to_hub if push_to_hub else get_random_name()

        # Use model or default from example config (zai-org/GLM-4.5)
        model_name = model or "zai-org/GLM-4.5"

        # Import url utilities
        from yourbench.utils.url_utils import get_api_key_for_url, validate_api_key_for_url

        # Special handling for OpenAI model names - automatically route to OpenAI
        # This is a superficial/temporary routing at the CLI level only
        openai_model_prefixes = ["gpt-4", "o1", "o3", "o4", "gpt-5"]
        if not base_url and any(model_name.startswith(prefix) for prefix in openai_model_prefixes):
            # Auto-route to OpenAI for these specific model names
            base_url = "https://api.openai.com/v1"
            logger.info(f"Auto-routing model '{model_name}' to OpenAI API")

        # Determine API key based on base_url if not explicitly provided
        if api_key:
            # Use the explicitly provided API key
            model_api_key = api_key
        else:
            # Determine API key based on URL
            model_api_key = get_api_key_for_url(base_url)

            # Validate that the required environment variable is set
            is_valid, error_msg = validate_api_key_for_url(base_url, model_api_key, model_name)
            if not is_valid:
                if base_url and model_api_key == "$HF_TOKEN":
                    # For custom base URLs with no specific API key, just warn
                    logger.warning(f"No specific API key for base URL '{base_url}'. Using HF_TOKEN if available.")
                else:
                    logger.error(error_msg)
                    raise typer.Exit(1)

        # Create model configuration with all CLI overrides
        model_config = ModelConfig(
            model_name=model_name,
            base_url=base_url,
            api_key=model_api_key,
            provider=provider,
            max_concurrent_requests=max_concurrent_requests or 32,
            encoding_name=encoding_name or "cl100k_base",
            bill_to=bill_to,
            extra_parameters=_parse_extra_parameters(model_extra_parameters),
        )

        # Set output directory
        local_output_dir = output_dir or Path("output")

        # Create HuggingFace configuration
        hf_config = HuggingFaceConfig(
            hf_dataset_name=dataset_name,
            hf_organization="$HF_ORGANIZATION",
            hf_token="$HF_TOKEN",  # This will be expanded from environment
            private=private_dataset,
            local_dataset_dir=local_output_dir / "dataset",
            local_saving=local_saving,
            export_jsonl=export_jsonl,
            jsonl_export_dir=jsonl_export_dir or Path.cwd() / "output",
        )

        # Prepare documents directory
        if config_or_docs_path.is_file():
            # Single file - create a directory and copy it
            raw_dir = local_output_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(config_or_docs_path, raw_dir)
        else:
            # Directory - use absolute path
            raw_dir = config_or_docs_path.resolve()

        # Create pipeline configuration with proper defaults
        # Set up ingestion config
        ingestion_config = IngestionConfig(
            run=True,
            source_documents_dir=raw_dir,
            output_dir=local_output_dir / "processed",
            upload_to_hub=True,  # Always upload to hub for pipeline to work
            llm_ingestion=llm_ingestion,
            pdf_dpi=pdf_dpi or 300,
        )

        # Set up summarization config
        summarization_config = SummarizationConfig(
            run=True,
            max_tokens=max_tokens or 32768,
            token_overlap=token_overlap or 512,
        )

        # Set up chunking config
        chunking_config = ChunkingConfig(
            run=True,
            l_max_tokens=l_max_tokens or 8192,
            token_overlap=token_overlap or 512,
            h_min=h_min or 2,
            h_max=h_max or 5,
        )

        # Set up question generation configs
        single_shot_config = SingleShotQuestionGenerationConfig(
            run=single_shot_questions if single_shot_questions is not None else True,
            question_mode=question_mode or "open-ended",
            additional_instructions=additional_instructions or "",
        )

        multi_hop_config = MultiHopQuestionGenerationConfig(
            run=multi_hop_questions,
            question_mode=question_mode or "open-ended",
            additional_instructions=additional_instructions or "",
        )

        cross_doc_config = CrossDocumentQuestionGenerationConfig(
            run=cross_doc_questions,
            question_mode=question_mode or "open-ended",
            additional_instructions=additional_instructions or "",
        )

        # Set up question rewriting config
        question_rewriting_config = QuestionRewritingConfig(
            run=question_rewriting,
        )

        pipeline_config = PipelineConfig(
            ingestion=ingestion_config,
            summarization=summarization_config,
            chunking=chunking_config,
            single_shot_question_generation=single_shot_config,
            multi_hop_question_generation=multi_hop_config,
            cross_document_question_generation=cross_doc_config,
            question_rewriting=question_rewriting_config,
            prepare_lighteval=LightevalConfig(run=True),
        )

        # Model roles mapping
        model_roles = {
            "ingestion": [model_name],
            "summarization": [model_name],
            "chunking": [model_name],
            "single_shot_question_generation": [model_name],
            "multi_hop_question_generation": [model_name],
            "cross_document_question_generation": [model_name],
            "question_rewriting": [model_name],
        }

        # Create final configuration
        config = YourbenchConfig(
            hf_configuration=hf_config,
            pipeline_config=pipeline_config,
            model_list=[model_config],
            model_roles=model_roles,
            debug=debug,
        )

        # Save config to temporary file for the pipeline
        config_path = local_output_dir / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(config_path)

        logger.info(f"Processing documents from: {config_or_docs_path}")
        logger.debug(f"Using model: {model_name}")
        logger.debug(f"Output directory: {local_output_dir}")
        if push_to_hub:
            logger.info(f"Will push to Hub as: {push_to_hub}")

        # Run the pipeline
        try:
            pipeline_func = _lazy_import_pipeline()
            pipeline_func(config_file_path=str(config_path), debug=debug)

            # Check for JSONL files
            jsonl_files = list(Path.cwd().glob("*.jsonl"))
            if jsonl_files:
                logger.success("Generated JSONL files:")
                for file in jsonl_files:
                    logger.info(f"  - {file.name}")

            if push_to_hub:
                logger.success(f"Dataset pushed to: https://huggingface.co/datasets/{push_to_hub}")

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            raise typer.Exit(1)


@app.command()
def run(
    config_or_docs: str = typer.Argument(..., help="Path to config file (YAML) or documents directory/file"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use (default: zai-org/GLM-4.5)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Base URL for the model API"),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key for the model (default: uses HF_TOKEN or OPENAI_API_KEY based on base_url)"
    ),
    model_extra_parameters: Optional[str] = typer.Option(
        None,
        "--model-extra-parameters",
        help=(
            "JSON string or path to JSON file with provider-specific payload options to include in chat completions"
        ),
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Provider for the model (e.g., openai, anthropic, auto)"
    ),
    single_shot_questions: Optional[bool] = typer.Option(
        None,
        "--single-shot-questions/--no-single-shot-questions",
        help="Generate single-shot questions (default: True)",
    ),
    multi_hop_questions: bool = typer.Option(
        False, "--multi-hop-questions", help="Generate multi-hop questions (default: False)"
    ),
    cross_doc_questions: bool = typer.Option(
        False, "--cross-doc-questions", help="Generate cross-document questions (default: False)"
    ),
    additional_instructions: Optional[str] = typer.Option(
        None, "--additional-instructions", "-i", help="Additional instructions for question generation"
    ),
    push_to_hub: Optional[str] = typer.Option(None, "--push-to-hub", help="Dataset name to push to HuggingFace Hub"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Local output directory"),
    # Additional pipeline configuration options
    question_mode: Optional[str] = typer.Option(
        None, "--question-mode", help="Question mode: open-ended or multi-choice (default: open-ended)"
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", help="Max tokens for summarization (default: 32768)"
    ),
    token_overlap: Optional[int] = typer.Option(
        None, "--token-overlap", help="Token overlap for chunking/summarization (default: 512)"
    ),
    l_max_tokens: Optional[int] = typer.Option(
        None, "--chunk-max-tokens", help="Max tokens per chunk (default: 8192)"
    ),
    h_min: Optional[int] = typer.Option(None, "--h-min", help="Min hop distance for multi-hop questions (default: 2)"),
    h_max: Optional[int] = typer.Option(None, "--h-max", help="Max hop distance for multi-hop questions (default: 5)"),
    pdf_dpi: Optional[int] = typer.Option(None, "--pdf-dpi", help="DPI for PDF processing (default: 300)"),
    llm_ingestion: bool = typer.Option(False, "--llm-ingestion", help="Use LLM for PDF ingestion (default: False)"),
    question_rewriting: bool = typer.Option(
        False, "--question-rewriting", help="Enable question rewriting (default: False)"
    ),
    private_dataset: bool = typer.Option(
        True, "--private/--public", help="Make dataset private on HuggingFace Hub (default: True)"
    ),
    local_saving: bool = typer.Option(
        True, "--save-local/--no-save-local", help="Save dataset locally (default: True)"
    ),
    export_jsonl: bool = typer.Option(
        True, "--export-jsonl/--no-export-jsonl", help="Export dataset as JSONL (default: True)"
    ),
    jsonl_export_dir: Optional[Path] = typer.Option(
        None, "--jsonl-export-dir", help="Directory for JSONL export (default: current directory)"
    ),
    max_concurrent_requests: Optional[int] = typer.Option(
        None, "--max-concurrent-requests", help="Max concurrent API requests (default: 32)"
    ),
    encoding_name: Optional[str] = typer.Option(
        None, "--encoding-name", help="Encoding name for tokenization (default: cl100k_base)"
    ),
    bill_to: Optional[str] = typer.Option(None, "--bill-to", help="Billing information for the model"),
) -> None:
    """YourBench - Generate Q&A pairs from documents or run with config file."""
    run_yourbench(
        config_or_docs=config_or_docs,
        model=model,
        base_url=base_url,
        api_key=api_key,
        model_extra_parameters=model_extra_parameters,
        provider=provider,
        single_shot_questions=single_shot_questions,
        multi_hop_questions=multi_hop_questions,
        cross_doc_questions=cross_doc_questions,
        additional_instructions=additional_instructions,
        push_to_hub=push_to_hub,
        debug=debug,
        output_dir=output_dir,
        question_mode=question_mode,
        max_tokens=max_tokens,
        token_overlap=token_overlap,
        l_max_tokens=l_max_tokens,
        h_min=h_min,
        h_max=h_max,
        pdf_dpi=pdf_dpi,
        llm_ingestion=llm_ingestion,
        question_rewriting=question_rewriting,
        private_dataset=private_dataset,
        local_saving=local_saving,
        export_jsonl=export_jsonl,
        jsonl_export_dir=jsonl_export_dir,
        max_concurrent_requests=max_concurrent_requests,
        encoding_name=encoding_name,
        bill_to=bill_to,
    )


@app.command("version")
def version_command() -> None:
    """Show YourBench version."""
    show_version()


def show_version() -> None:
    """Display version information."""
    from importlib.metadata import version as get_version

    try:
        v = get_version("yourbench")
        print(f"YourBench version: {v}")
    except Exception:
        print("YourBench version: development")


def main() -> None:
    """Entry point for the CLI that handles special cases."""
    import sys

    # Handle version flag specially
    if "--version" in sys.argv or "-v" in sys.argv:
        show_version()
        return

    # For backward compatibility, if no arguments provided, show help
    if len(sys.argv) == 1:
        app()
        return

    # Check if first real argument (after script name) looks like a path
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        # If first arg is an option, use normal processing
        if first_arg.startswith("-"):
            app()
            return

        # If we have a path argument (config file or data directory)
        # directly call the run function without the 'run' command
        # This allows: yourbench config.yaml or yourbench data_dir
        if first_arg not in ["run", "version"]:
            # Insert 'run' command to make it work with typer
            new_args = [sys.argv[0], "run"] + sys.argv[1:]
            sys.argv = new_args

    app()


if __name__ == "__main__":
    main()
