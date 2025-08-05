#!/usr/bin/env python3
"""YourBench CLI - Dynamic Evaluation Set Generation with Large Language Models."""

from __future__ import annotations
import os
import sys
import time
from typing import Optional
from pathlib import Path


# Track startup time
startup_time = time.perf_counter()

import typer
from dotenv import load_dotenv
from loguru import logger
from randomname import get_name as get_random_name


# Early startup logging
logger.info("YourBench starting up...")
logger.info("Loading core modules...")

# Lazy imports - only import when needed
run_pipeline = None


def _lazy_import_pipeline():
    global run_pipeline
    if run_pipeline is None:
        logger.info("Loading pipeline components...")
        from yourbench.pipeline.handler import run_pipeline as _run_pipeline

        run_pipeline = _run_pipeline
    return run_pipeline


logger.info("Loading environment variables...")
load_dotenv()

app = typer.Typer(
    name="yourbench",
    help="YourBench - Dynamic Evaluation Set Generation with Large Language Models.",
    pretty_exceptions_show_locals=False,
)

# Log startup completion
logger.success(f"YourBench loaded in {time.perf_counter() - startup_time:.2f}s")


@app.command()
def run(
    config_or_docs: Path = typer.Argument(..., help="Path to config file (YAML) or documents directory/file"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use (overrides config if set)"),
    single_shot_questions: Optional[bool] = typer.Option(
        None, "--single-shot-questions/--no-single-shot-questions", help="Generate single-shot questions"
    ),
    additional_instructions: Optional[str] = typer.Option(
        None, "--additional-instructions", "-i", help="Additional instructions for question generation"
    ),
    push_to_hub: Optional[str] = typer.Option(None, "--push-to-hub", help="Dataset name to push to HuggingFace Hub"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Local output directory"),
) -> None:
    """Run YourBench pipeline with config file or generate Q&A pairs from documents."""
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if debug else "INFO")

    # Check if input is a YAML config file
    if config_or_docs.suffix in [".yaml", ".yml"] and config_or_docs.exists():
        logger.info(f"Running with config file: {config_or_docs}")

        # Run pipeline with existing config
        pipeline_func = _lazy_import_pipeline()
        try:
            pipeline_func(config_file_path=str(config_or_docs), debug=debug)
        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            raise typer.Exit(1)
    else:
        # Generate config dynamically for document path
        if not config_or_docs.exists():
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
            SingleShotQuestionGenerationConfig,
        )

        # Create configuration programmatically
        dataset_name = push_to_hub if push_to_hub else get_random_name()

        # Use model or default
        model_name = model or "gpt-4o-mini"

        # Determine model configuration
        model_config = ModelConfig(model_name=model_name)
        if "gpt" in model_name.lower():
            model_config.base_url = "https://api.openai.com/v1/"
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
            hf_token="$HF_TOKEN",
            private=True,
            local_dataset_dir=local_output_dir / "dataset",
            local_saving=True,
            export_jsonl=True,
            jsonl_export_dir=Path.cwd(),
        )

        # Prepare documents directory
        if config_or_docs.is_file():
            # Single file - create a directory and copy it
            raw_dir = local_output_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(config_or_docs, raw_dir)
        else:
            # Directory - use absolute path
            raw_dir = config_or_docs.resolve()

        # Create pipeline configuration
        pipeline_config = PipelineConfig(
            ingestion=IngestionConfig(
                run=True,
                source_documents_dir=raw_dir,
                output_dir=local_output_dir / "processed",
                upload_to_hub=bool(push_to_hub),
            ),
            summarization=SummarizationConfig(run=True),
            chunking=ChunkingConfig(run=True),
            single_shot_question_generation=SingleShotQuestionGenerationConfig(
                run=single_shot_questions if single_shot_questions is not None else True,
                additional_instructions=additional_instructions or "Generate questions to test a curious adult",
            ),
            prepare_lighteval=LightevalConfig(run=True),
        )

        # Model roles mapping
        model_roles = {
            "ingestion": [model_name],
            "summarization": [model_name],
            "chunking": [model_name],
            "single_shot_question_generation": [model_name],
            "multi_hop_question_generation": [model_name],
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

        logger.info(f"Running YourBench with model: {model_name}")
        logger.info(f"Processing documents from: {config_or_docs}")
        if push_to_hub:
            logger.info(f"Will push dataset to Hub as: {push_to_hub}")

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
def version() -> None:
    """Show YourBench version."""
    from importlib.metadata import version as get_version

    try:
        v = get_version("yourbench")
        logger.info(f"YourBench version: {v}")
    except Exception:
        logger.info("YourBench version: development")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
