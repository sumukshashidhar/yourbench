#!/usr/bin/env python3
"""YourBench CLI - Dynamic Evaluation Set Generation with Large Language Models."""

from __future__ import annotations
import os
import sys
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


def run_yourbench(
    config_or_docs: str,
    model: Optional[str] = None,
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

        # Determine model configuration
        model_config = ModelConfig(model_name=model_name, max_concurrent_requests=max_concurrent_requests or 32)
        if "gpt" in model_name.lower():
            model_config.api_key = "$OPENAI_API_KEY"
            if not os.getenv("OPENAI_API_KEY"):
                logger.warning("OPENAI_API_KEY not set. Please set it with: export OPENAI_API_KEY=your_key")
        else:
            model_config.api_key = "$HF_TOKEN"
            if not os.getenv("HF_TOKEN"):
                logger.warning("HF_TOKEN not set. Please set it with: export HF_TOKEN=your_token")

        # Set output directory
        local_output_dir = output_dir or Path("yourbench_output")

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
) -> None:
    """YourBench - Generate Q&A pairs from documents or run with config file."""
    run_yourbench(
        config_or_docs=config_or_docs,
        model=model,
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
