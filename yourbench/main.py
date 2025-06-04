#!/usr/bin/env python3
"""YourBench CLI - Dynamic Evaluation Set Generation with Large Language Models."""

from __future__ import annotations
import sys
from typing import Optional
from pathlib import Path
from dataclasses import field, dataclass

import yaml
import typer
from loguru import logger
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.console import Console

from yourbench.analysis import run_analysis
from yourbench.pipeline.handler import run_pipeline


app = typer.Typer(
    name="yourbench",
    help="YourBench - Dynamic Evaluation Set Generation with Large Language Models.",
    pretty_exceptions_show_locals=False,
)
console = Console()


@dataclass
class ConfigBuilder:
    """Builder for creating YourBench configuration files."""

    # HF Configuration
    hf_dataset_name: str = ""
    hf_organization: str = "$HF_ORGANIZATION"
    hf_token: str = "$HF_TOKEN"
    private: bool = False
    concat_if_exist: bool = False
    local_dataset_dir: Optional[str] = None
    local_saving: bool = False

    # Models and pipeline
    models: list[dict] = field(default_factory=list)
    model_roles: dict = field(default_factory=dict)
    pipeline_config: dict = field(default_factory=dict)

    def build(self) -> dict:
        """Build the configuration dictionary."""
        config = {
            "hf_configuration": {
                "hf_dataset_name": self.hf_dataset_name,
                "hf_organization": self.hf_organization,
                "token": self.hf_token,
                "private": self.private,
                "concat_if_exist": self.concat_if_exist,
            }
        }

        # Add local dataset options if configured
        if self.local_dataset_dir:
            config["hf_configuration"]["local_dataset_dir"] = self.local_dataset_dir
            config["hf_configuration"]["local_saving"] = self.local_saving

        config["model_list"] = self.models

        if self.model_roles:
            config["model_roles"] = self.model_roles

        config["pipeline"] = self.pipeline_config

        return config


def create_model_config(existing_models: list[str]) -> dict:
    """Interactive model configuration with smart provider logic."""
    console.print("\n[bold cyan]Model Configuration[/bold cyan]")

    model_name = Prompt.ask("Model name", default="Qwen/Qwen3-30B-A3B")

    # Provider type selection
    console.print("\nSelect inference type:")
    console.print("1. Hugging Face Inference (default)")
    console.print("2. OpenAI Compatible API (vLLM, etc.)")
    console.print("3. Custom API endpoint")

    choice = IntPrompt.ask("Choice", default=1)

    config = {"model_name": model_name}

    if choice == 1:  # Hugging Face
        # Only for HF, ask about provider
        if Confirm.ask("Use a specific provider?", default=False):
            console.print("\nAvailable providers:")
            console.print("- fireworks-ai")
            console.print("- together-ai")
            console.print("- deepinfra")
            console.print("- huggingface (default)")
            provider = Prompt.ask("Provider", default="huggingface")
            if provider != "huggingface":
                config["provider"] = provider

    elif choice == 2:  # OpenAI Compatible
        config["base_url"] = "https://api.openai.com/v1"
        config["api_key"] = Prompt.ask("API key (use $VAR for env)", default="$OPENAI_API_KEY")
        config["model_name"] = Prompt.ask("Model name", default="gpt-4o")

    else:  # Custom
        config["base_url"] = Prompt.ask("Base URL")
        config["api_key"] = Prompt.ask("API key (use $VAR for env)", default="$API_KEY")

    # Advanced options
    if Confirm.ask("\nConfigure advanced options?", default=False):
        config["max_concurrent_requests"] = IntPrompt.ask("Max concurrent requests", default=16)
        if Confirm.ask("Use custom tokenizer?", default=False):
            config["encoding_name"] = Prompt.ask("Encoding name", default="cl100k_base")
    else:
        config["max_concurrent_requests"] = 16 if choice == 1 else 8

    return config


def configure_model_roles(models: list[dict]) -> dict:
    """Configure which models to use for each pipeline stage."""
    if not models:
        return {}

    if len(models) == 1:
        # Single model - use for everything
        model_name = models[0]["model_name"]
        return {
            "ingestion": [model_name],
            "summarization": [model_name],
            "chunking": ["intfloat/multilingual-e5-large-instruct"],
            "single_shot_question_generation": [model_name],
            "multi_hop_question_generation": [model_name],
        }

    console.print("\n[bold cyan]Model Role Assignment[/bold cyan]")
    console.print("Assign models to pipeline stages:")

    # Show available models
    table = Table(title="Available Models")
    table.add_column("Index", style="cyan")
    table.add_column("Model", style="green")
    for i, model in enumerate(models, 1):
        table.add_row(str(i), model["model_name"])
    console.print(table)

    roles = {}
    stages = [
        ("ingestion", "Document parsing & conversion"),
        ("summarization", "Document summarization"),
        ("single_shot_question_generation", "Single-hop questions"),
        ("multi_hop_question_generation", "Multi-hop questions"),
    ]

    for stage, desc in stages:
        console.print(f"\n[yellow]{stage}[/yellow]: {desc}")
        indices = Prompt.ask("Model indices (comma-separated, e.g., 1,2)", default="1")
        selected = []
        for idx in indices.split(","):
            try:
                i = int(idx.strip()) - 1
                if 0 <= i < len(models):
                    selected.append(models[i]["model_name"])
            except ValueError:
                pass
        if selected:
            roles[stage] = selected

    # Chunking always uses embedding model
    roles["chunking"] = ["intfloat/multilingual-e5-large-instruct"]

    return roles


def configure_ingestion(enabled: bool) -> dict:
    """Configure ingestion stage."""
    config = {"run": enabled}

    if not enabled:
        return config

    if Confirm.ask("\nConfigure ingestion paths?", default=False):
        config["source_documents_dir"] = Prompt.ask("Source documents directory", default="data/raw")
        config["output_dir"] = Prompt.ask("Output directory", default="data/processed")
    else:
        config["source_documents_dir"] = "data/raw"
        config["output_dir"] = "data/processed"

    return config


def configure_summarization(enabled: bool) -> dict:
    """Configure summarization stage."""
    config = {"run": enabled}

    if not enabled:
        return config

    if Confirm.ask("\nConfigure summarization options?", default=False):
        config["max_tokens"] = IntPrompt.ask("Max tokens per chunk", default=16384)
        config["token_overlap"] = IntPrompt.ask("Token overlap", default=128)
        config["encoding_name"] = Prompt.ask("Tokenizer encoding", default="cl100k_base")

    return config


def configure_chunking(enabled: bool) -> dict:
    """Configure chunking stage."""
    config = {"run": enabled}

    if not enabled:
        return config

    if Confirm.ask("\nConfigure chunking parameters?", default=False):
        chunk_config = {}
        chunk_config["l_max_tokens"] = IntPrompt.ask("Max tokens per chunk", default=256)
        chunk_config["token_overlap"] = IntPrompt.ask("Token overlap", default=0)
        chunk_config["encoding_name"] = Prompt.ask("Tokenizer encoding", default="cl100k_base")

        # Multi-hop configuration
        if Confirm.ask("Configure multi-hop parameters?", default=True):
            chunk_config["h_min"] = IntPrompt.ask("Min chunks for multi-hop", default=2)
            chunk_config["h_max"] = IntPrompt.ask("Max chunks for multi-hop", default=5)
            chunk_config["num_multihops_factor"] = IntPrompt.ask("Multi-hop factor", default=2)

        config["chunking_configuration"] = chunk_config
    else:
        config["chunking_configuration"] = {
            "l_max_tokens": 256,
            "token_overlap": 0,
            "encoding_name": "cl100k_base",
            "h_min": 2,
            "h_max": 5,
            "num_multihops_factor": 2,
        }

    return config


def configure_question_generation(stage_name: str, enabled: bool) -> dict:
    """Configure question generation stages."""
    config = {"run": enabled}

    if not enabled:
        return config

    if Confirm.ask(f"\nConfigure {stage_name.replace('_', ' ')} options?", default=False):
        # Question type
        console.print("\nQuestion type:")
        console.print("1. Open-ended (default)")
        console.print("2. Multiple choice")
        q_type = IntPrompt.ask("Choice", default=1)
        config["question_type"] = "multi-choice" if q_type == 2 else "open-ended"

        # Additional instructions
        config["additional_instructions"] = Prompt.ask(
            "Additional instructions", default="Generate questions to test a curious adult"
        )

        # Chunk sampling
        if Confirm.ask("Configure chunk sampling?", default=False):
            sampling = {}
            console.print("\nSampling mode:")
            console.print("1. All chunks (default)")
            console.print("2. Sample by count")
            console.print("3. Sample by percentage")
            mode = IntPrompt.ask("Choice", default=1)

            if mode == 2:
                sampling["mode"] = "count"
                sampling["value"] = IntPrompt.ask("Number of chunks", default=10)
            elif mode == 3:
                sampling["mode"] = "percentage"
                sampling["value"] = FloatPrompt.ask("Percentage (0-1)", default=0.5)
            else:
                sampling["mode"] = "all"

            if mode in [2, 3]:
                sampling["random_seed"] = IntPrompt.ask("Random seed", default=42)

            config["chunk_sampling"] = sampling

    return config


def configure_pipeline_stages() -> dict:
    """Configure all pipeline stages with cascading options."""
    console.print("\n[bold cyan]Pipeline Configuration[/bold cyan]")

    stages = {
        "ingestion": ("Convert documents to markdown", configure_ingestion),
        "upload_ingest_to_hub": ("Upload to Hugging Face Hub", lambda x: {"run": x}),
        "summarization": ("Generate document summaries", configure_summarization),
        "chunking": ("Split documents into chunks", configure_chunking),
        "single_shot_question_generation": ("Generate single-hop questions", configure_question_generation),
        "multi_hop_question_generation": ("Generate multi-hop questions", configure_question_generation),
        "lighteval": ("Create evaluation dataset", lambda x: {"run": x}),
        "citation_score_filtering": ("Add citation scores", lambda x: {"run": x}),
    }

    pipeline_config = {}

    # First, ask which stages to enable
    console.print("Select pipeline stages to enable:")
    enabled_stages = {}

    for stage, (desc, _) in stages.items():
        enabled = Confirm.ask(f"  {stage} - {desc}", default=True)
        enabled_stages[stage] = enabled

    # Then configure each enabled stage
    for stage, (desc, configure_fn) in stages.items():
        if stage in ["single_shot_question_generation", "multi_hop_question_generation"]:
            pipeline_config[stage] = configure_fn(stage, enabled_stages[stage])
        else:
            pipeline_config[stage] = configure_fn(enabled_stages[stage])

    return pipeline_config


@app.command()
def run(
    config_path: Optional[Path] = typer.Argument(
        None,
        help="Path to configuration file (YAML/JSON)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="[LEGACY] Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    plot_stage_timing: bool = typer.Option(
        False,
        "--plot-stage-timing",
        help="Generate stage timing chart",
    ),
) -> None:
    """Run the YourBench pipeline with a configuration file."""
    # Handle both new positional and legacy --config
    final_config = config_path or config

    if not final_config:
        console.print("[red]Error:[/red] Please provide a configuration file")
        console.print("Usage: yourbench run CONFIG_FILE")
        raise typer.Exit(1)

    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if debug else "INFO")

    logger.info(f"Running pipeline with config: {final_config}")

    try:
        run_pipeline(
            config_file_path=str(final_config),
            debug=debug,
            plot_stage_timing=plot_stage_timing,
        )
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise typer.Exit(1)


@app.command()
def create(
    output: Path = typer.Argument(
        "config.yaml",
        help="Output configuration file path",
    ),
    simple: bool = typer.Option(
        False,
        "--simple",
        "-s",
        help="Create a simple configuration with minimal options",
    ),
) -> None:
    """Create a new YourBench configuration file interactively."""
    console.print("[bold green]YourBench Configuration Creator[/bold green]\n")

    builder = ConfigBuilder()

    # Basic configuration
    console.print("[bold cyan]Basic Configuration[/bold cyan]")
    builder.hf_dataset_name = Prompt.ask("Hugging Face dataset name", default="my-yourbench-dataset")

    if simple:
        # Simple mode - minimal questions
        builder.models = [
            {
                "model_name": "Qwen/Qwen3-30B-A3B",
                "provider": "fireworks-ai",
            }
        ]
        builder.pipeline_config = {
            "ingestion": {
                "run": True,
                "source_documents_dir": "data/raw",
                "output_dir": "data/processed",
            },
            "upload_ingest_to_hub": {"run": True},
            "summarization": {"run": True},
            "chunking": {"run": True},
            "single_shot_question_generation": {"run": True},
            "multi_hop_question_generation": {"run": True},
            "lighteval": {"run": True},
            "citation_score_filtering": {"run": True},
        }
    else:
        # Advanced configuration
        if Confirm.ask("Configure Hugging Face options?", default=True):
            builder.hf_organization = Prompt.ask("Organization (use $VAR for env)", default="$HF_ORGANIZATION")
            builder.hf_token = Prompt.ask("HF Token (use $VAR for env)", default="$HF_TOKEN")
            builder.private = Confirm.ask("Make dataset private?", default=False)
            builder.concat_if_exist = Confirm.ask("Concatenate if dataset exists?", default=False)

            # Local dataset options
            if Confirm.ask("Configure local dataset storage?", default=False):
                builder.local_dataset_dir = Prompt.ask("Local dataset directory", default="data/datasets")
                builder.local_saving = Confirm.ask("Save dataset locally?", default=True)

        # Model configuration
        console.print("\n[bold cyan]Model Configuration[/bold cyan]")
        add_models = Confirm.ask("Add models?", default=True)

        while add_models:
            model = create_model_config([m["model_name"] for m in builder.models])
            builder.models.append(model)
            add_models = Confirm.ask("\nAdd another model?", default=False)

        # Model roles
        if len(builder.models) > 1 and Confirm.ask("\nAssign models to specific stages?", default=True):
            builder.model_roles = configure_model_roles(builder.models)
        elif builder.models:
            # Single model or default assignment
            model_name = builder.models[0]["model_name"]
            builder.model_roles = {
                "ingestion": [model_name],
                "summarization": [model_name],
                "chunking": ["intfloat/multilingual-e5-large-instruct"],
                "single_shot_question_generation": [model_name],
                "multi_hop_question_generation": [model_name],
            }

        # Pipeline stages
        builder.pipeline_config = configure_pipeline_stages()

    # Build and save configuration
    config = builder.build()

    # Write to file
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, width=120)

    console.print(f"\n[green]✓[/green] Configuration saved to: {output}")

    # Show next steps
    console.print("\n[bold]Next steps:[/bold]")
    if config["pipeline"].get("ingestion", {}).get("run", False):
        src_dir = config["pipeline"]["ingestion"].get("source_documents_dir", "data/raw")
        console.print(f"1. Place your documents in: {src_dir}")
    console.print(f"2. Run: [cyan]yourbench run {output}[/cyan]")


@app.command()
def analyze(
    analysis_name: str = typer.Argument(..., help="Name of the analysis to run"),
    args: list[str] = typer.Argument(None, help="Additional arguments"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Run a specific analysis by name."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if debug else "INFO")

    logger.info(f"Running analysis '{analysis_name}' with arguments: {args}")

    try:
        run_analysis(analysis_name, args, debug=debug)
    except Exception as e:
        logger.exception(f"Analysis '{analysis_name}' failed: {e}")
        raise typer.Exit(1)


@app.command()
def gui() -> None:
    """Launch the Gradio UI (not yet implemented)."""
    logger.error("GUI support is not yet implemented")
    raise typer.Exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
