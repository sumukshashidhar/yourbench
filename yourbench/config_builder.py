"""Modern configuration builder using configuration_engine dataclasses."""

from __future__ import annotations
import json
from typing import Any
from pathlib import Path

import yaml
from loguru import logger
from randomname import get_name as get_random_name
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.console import Console

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
    CitationScoreFilteringConfig,
    MultiHopQuestionGenerationConfig,
    SingleShotQuestionGenerationConfig,
    CrossDocumentQuestionGenerationConfig,
)


console = Console()

# Configuration constants
DEFAULT_CONCURRENT_REQUESTS_HF = 16
DEFAULT_CONCURRENT_REQUESTS_API = 8
DEFAULT_CHUNK_TOKENS = 256
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TOKEN_OVERLAP = 128
DEFAULT_H_MIN = 2
DEFAULT_H_MAX = 5
DEFAULT_MULTIHOP_FACTOR = 2


def validate_api_key_format(api_key: str) -> tuple[bool, str]:
    """Validate API key format - should be env variable or empty."""
    if not api_key or api_key.startswith("$"):
        return True, api_key

    # Check for suspicious patterns
    suspicious_patterns = ["sk-", "key-", "api-", "hf_"]
    if len(api_key) > 10 and any(pattern in api_key for pattern in suspicious_patterns):
        return False, "Please use environment variable format (e.g., $OPENAI_API_KEY)"

    return True, api_key


def write_env_file(api_keys: dict[str, str]) -> None:
    """Write API keys to .env file if they don't exist."""
    env_path = Path(".env")

    # Read existing variables
    existing_vars = {}
    if env_path.exists():
        try:
            content = env_path.read_text()
            existing_vars = {
                key.strip(): value.strip()
                for line in content.splitlines()
                if (stripped_line := line.strip()) and not stripped_line.startswith("#") and "=" in stripped_line
                for key, value in [stripped_line.split("=", 1)]
            }
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not read .env file: {e}")
            return

    # Filter new keys
    new_keys = [(var, val) for var, val in api_keys.items() if var not in existing_vars]

    if not new_keys:
        return

    try:
        with env_path.open("a") as f:
            if existing_vars:  # Add newline if file has content
                f.write("\n")
            f.write("# API Keys added by YourBench\n")
            for var, val in new_keys:
                f.write(f"{var}={val}\n")
        console.print(f"[green]✓[/green] Added {len(new_keys)} API key(s) to .env file")
        console.print("[yellow]Remember to update .env with your actual API keys![/yellow]")
    except (OSError, PermissionError) as e:
        logger.error(f"Could not write to .env file: {e}")
        console.print("[red]Warning: Could not create/update .env file. Add these manually:[/red]")
        for var, val in new_keys:
            console.print(f"  {var}={val}")


def create_model_config(existing_models: list[str]) -> ModelConfig:
    """Interactive model configuration with smart provider logic."""
    console.print("\n[bold cyan]Model Configuration[/bold cyan]")

    model_name = Prompt.ask("Model name", default="Qwen/Qwen3-30B-A3B")

    # Provider type selection
    console.print("\nSelect inference type:")
    console.print("1. Hugging Face Inference (default)")
    console.print("2. OpenAI Compatible API (vLLM, etc.)")
    console.print("3. OpenAI API")
    console.print("4. Google Gemini API")
    console.print("5. Custom API endpoint")

    choice = IntPrompt.ask("Choice", default=1)

    api_keys_to_env = {}

    if choice == 1:  # Hugging Face
        provider = None
        if Confirm.ask("Use a specific provider?", default=False):
            console.print("\nAvailable providers:")
            console.print("- fireworks-ai")
            console.print("- together-ai")
            console.print("- deepinfra")
            console.print("- huggingface (default)")
            provider = Prompt.ask("Provider", default="huggingface")
            if provider == "huggingface":
                provider = None

        config = ModelConfig(
            model_name=model_name,
            provider=provider,
            api_key="$HF_TOKEN",
            max_concurrent_requests=DEFAULT_CONCURRENT_REQUESTS_HF,
        )
        api_keys_to_env["HF_TOKEN"] = "hf_..."

    elif choice == 2:  # OpenAI Compatible
        base_url = Prompt.ask("Base URL", default="http://localhost:8000/v1")
        while True:
            api_key = Prompt.ask("API key (use $VAR for env variables)", default="$VLLM_API_KEY")
            valid, msg = validate_api_key_format(api_key)
            if valid:
                break
            console.print(f"[red]Error: {msg}[/red]")

        config = ModelConfig(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            max_concurrent_requests=DEFAULT_CONCURRENT_REQUESTS_API,
        )
        if api_key.startswith("$"):
            api_keys_to_env[api_key[1:]] = "your-vllm-api-key-here"

    elif choice == 3:  # OpenAI
        while True:
            api_key = Prompt.ask("API key (use $VAR for env variables)", default="$OPENAI_API_KEY")
            valid, msg = validate_api_key_format(api_key)
            if valid:
                break
            console.print(f"[red]Error: {msg}[/red]")

        config = ModelConfig(
            model_name=Prompt.ask("Model name", default="gpt-4"),
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            max_concurrent_requests=DEFAULT_CONCURRENT_REQUESTS_API,
        )
        if api_key.startswith("$"):
            api_keys_to_env[api_key[1:]] = "sk-..."

    elif choice == 4:  # Gemini
        while True:
            api_key = Prompt.ask("API key (use $VAR for env variables)", default="$GEMINI_API_KEY")
            valid, msg = validate_api_key_format(api_key)
            if valid:
                break
            console.print(f"[red]Error: {msg}[/red]")

        config = ModelConfig(
            model_name=Prompt.ask("Model name", default="gemini-2.5-flash-preview"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
            max_concurrent_requests=DEFAULT_CONCURRENT_REQUESTS_API,
        )
        if api_key.startswith("$"):
            api_keys_to_env[api_key[1:]] = "your-gemini-api-key-here"

    else:  # Custom
        base_url = Prompt.ask("Base URL")
        while True:
            api_key = Prompt.ask("API key (use $VAR for env variables)", default="$API_KEY")
            valid, msg = validate_api_key_format(api_key)
            if valid:
                break
            console.print(f"[red]Error: {msg}[/red]")

        config = ModelConfig(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            max_concurrent_requests=DEFAULT_CONCURRENT_REQUESTS_API,
        )
        if api_key.startswith("$"):
            api_keys_to_env[api_key[1:]] = "your-api-key-here"

    # Write API keys to .env if needed
    if api_keys_to_env:
        write_env_file(api_keys_to_env)

    # Advanced options
    if Confirm.ask("\nConfigure advanced options?", default=False):
        config.max_concurrent_requests = IntPrompt.ask(
            "Max concurrent requests", default=config.max_concurrent_requests
        )
        if Confirm.ask("Use custom tokenizer?", default=False):
            config.encoding_name = Prompt.ask("Encoding name", default="cl100k_base")

    if Confirm.ask("Add custom provider parameters as JSON?", default=False):
        raw_params = Prompt.ask('Enter JSON object (e.g. {"reasoning": {"effort": "medium"}})', default="{}")
        try:
            parsed = json.loads(raw_params)
            if not isinstance(parsed, dict):
                raise ValueError("Extra parameters must be a JSON object")
            config.extra_parameters = parsed
        except (json.JSONDecodeError, ValueError) as exc:
            console.print(f"[red]Invalid JSON for extra parameters: {exc}. Ignoring input.[/red]")

    return config


def configure_model_roles(models: list[ModelConfig]) -> dict[str, list[str]]:
    """Configure which models to use for each pipeline stage."""
    if not models:
        return {}

    if len(models) == 1:
        # Single model - use for everything
        model_name = models[0].model_name
        return {
            "ingestion": [model_name] if model_name else [],
            "summarization": [model_name] if model_name else [],
            "single_shot_question_generation": [model_name] if model_name else [],
            "multi_hop_question_generation": [model_name] if model_name else [],
            "cross_document_question_generation": [model_name] if model_name else [],
            "question_rewriting": [model_name] if model_name else [],
        }

    console.print("\n[bold cyan]Model Role Assignment[/bold cyan]")
    console.print("Assign models to pipeline stages:")

    # Show available models
    table = Table(title="Available Models")
    table.add_column("Index", style="cyan")
    table.add_column("Model", style="green")
    for i, model in enumerate(models, 1):
        table.add_row(str(i), model.model_name)
    console.print(table)

    roles = {}
    stages = [
        ("ingestion", "Document parsing & conversion"),
        ("summarization", "Document summarization"),
        ("single_shot_question_generation", "Single-hop questions"),
        ("multi_hop_question_generation", "Multi-hop questions"),
        ("cross_document_question_generation", "Cross-document questions"),
        ("question_rewriting", "Question rewriting"),
    ]

    for stage, desc in stages:
        console.print(f"\n[yellow]{stage}[/yellow]: {desc}")
        indices = Prompt.ask("Model indices (comma-separated, e.g., 1,2)", default="1")
        selected = []
        for idx in indices.split(","):
            try:
                i = int(idx.strip()) - 1
                if 0 <= i < len(models):
                    model_name = models[i].model_name
                    if model_name:
                        selected.append(model_name)
                else:
                    logger.warning(f"Model index {idx} is out of range (1-{len(models)})")
            except ValueError:
                logger.warning(f"Invalid model index '{idx}' - expected a number")
        if selected:
            roles[stage] = selected

    return roles


def configure_pipeline_stages() -> PipelineConfig:
    """Configure all pipeline stages with modern dataclass approach."""
    console.print("\n[bold cyan]Pipeline Configuration[/bold cyan]")

    # Ask which stages to enable
    console.print("Select pipeline stages to enable:")

    # Ingestion
    ingestion_enabled = Confirm.ask("  ingestion - Convert documents to markdown", default=True)
    ingestion_config = IngestionConfig(run=ingestion_enabled)
    if ingestion_enabled and Confirm.ask("Configure ingestion paths?", default=False):
        ingestion_config.source_documents_dir = Path(Prompt.ask("Source documents directory", default="data/raw"))
        ingestion_config.output_dir = Path(Prompt.ask("Output directory", default="data/processed"))

    # Summarization
    summarization_enabled = Confirm.ask("  summarization - Generate document summaries", default=True)
    summarization_config = SummarizationConfig(run=summarization_enabled)
    if summarization_enabled and Confirm.ask("Configure summarization options?", default=False):
        summarization_config.max_tokens = IntPrompt.ask("Max tokens per chunk", default=DEFAULT_MAX_TOKENS)
        summarization_config.token_overlap = IntPrompt.ask("Token overlap", default=DEFAULT_TOKEN_OVERLAP)
        summarization_config.encoding_name = Prompt.ask("Tokenizer encoding", default="cl100k_base")

    # Chunking
    chunking_enabled = Confirm.ask("  chunking - Split documents into chunks", default=True)
    chunking_config = ChunkingConfig(run=chunking_enabled)
    if chunking_enabled and Confirm.ask("Configure chunking parameters?", default=False):
        chunking_config.l_max_tokens = IntPrompt.ask("Max tokens per chunk", default=DEFAULT_CHUNK_TOKENS)
        chunking_config.token_overlap = IntPrompt.ask("Token overlap", default=0)
        chunking_config.encoding_name = Prompt.ask("Tokenizer encoding", default="cl100k_base")

        # Multi-hop configuration
        if Confirm.ask("Configure multi-hop parameters?", default=True):
            chunking_config.h_min = IntPrompt.ask("Min chunks for multi-hop", default=DEFAULT_H_MIN)
            chunking_config.h_max = IntPrompt.ask("Max chunks for multi-hop", default=DEFAULT_H_MAX)
            chunking_config.num_multihops_factor = IntPrompt.ask("Multi-hop factor", default=DEFAULT_MULTIHOP_FACTOR)

    # Single-shot question generation
    single_shot_enabled = Confirm.ask(
        "  single_shot_question_generation - Generate single-hop questions", default=True
    )
    single_shot_config = SingleShotQuestionGenerationConfig(run=single_shot_enabled)

    # Multi-hop question generation
    multi_hop_enabled = Confirm.ask("  multi_hop_question_generation - Generate multi-hop questions", default=True)
    multi_hop_config = MultiHopQuestionGenerationConfig(run=multi_hop_enabled)

    # Cross-document question generation
    cross_doc_enabled = Confirm.ask(
        "  cross_document_question_generation - Generate cross-document questions", default=False
    )
    cross_doc_config = CrossDocumentQuestionGenerationConfig(run=cross_doc_enabled)
    if cross_doc_enabled and Confirm.ask("Configure cross-document options?", default=False):
        cross_doc_config.max_combinations = IntPrompt.ask("Max combinations", default=100)
        cross_doc_config.chunks_per_document = IntPrompt.ask("Chunks per document", default=1)
        console.print("Number of documents per combination (comma-separated):")
        docs_input = Prompt.ask("E.g., 2,3,5", default="2,5")
        cross_doc_config.num_docs_per_combination = [int(x.strip()) for x in docs_input.split(",")]

    # Question rewriting
    rewriting_enabled = Confirm.ask("  question_rewriting - Rewrite questions", default=False)
    rewriting_config = QuestionRewritingConfig(run=rewriting_enabled)

    # LightEval preparation
    prepare_lighteval_enabled = Confirm.ask("  prepare_lighteval - Prepare evaluation dataset", default=True)
    prepare_lighteval_config = LightevalConfig(run=prepare_lighteval_enabled)

    # Citation score filtering
    citation_enabled = Confirm.ask("  citation_score_filtering - Add citation scores", default=True)
    citation_config = CitationScoreFilteringConfig(run=citation_enabled)

    return PipelineConfig(
        ingestion=ingestion_config,
        summarization=summarization_config,
        chunking=chunking_config,
        single_shot_question_generation=single_shot_config,
        multi_hop_question_generation=multi_hop_config,
        cross_document_question_generation=cross_doc_config,
        question_rewriting=rewriting_config,
        prepare_lighteval=prepare_lighteval_config,
        citation_score_filtering=citation_config,
    )


def create_yourbench_config(simple: bool = False) -> YourbenchConfig:
    """Create YourBench configuration interactively."""
    console.print("[bold green]YourBench Configuration Creator[/bold green]\n")

    if simple:
        # Simple mode - no prompts, use defaults
        hf_dataset_name = get_random_name()
        hf_config = HuggingFaceConfig(hf_dataset_name=hf_dataset_name)
        models = [
            ModelConfig(
                model_name="Qwen/Qwen3-30B-A3B",
                provider="fireworks-ai",
                api_key="$HF_TOKEN",
                max_concurrent_requests=DEFAULT_CONCURRENT_REQUESTS_HF,
            )
        ]
        pipeline_config = PipelineConfig(
            ingestion=IngestionConfig(run=True),
            summarization=SummarizationConfig(run=True),
            chunking=ChunkingConfig(run=True),
            single_shot_question_generation=SingleShotQuestionGenerationConfig(run=True),
            multi_hop_question_generation=MultiHopQuestionGenerationConfig(run=True),
            prepare_lighteval=LightevalConfig(run=True),
            citation_score_filtering=CitationScoreFilteringConfig(run=True),
        )
        model_roles = {}
    else:
        # Advanced configuration
        console.print("[bold cyan]Hugging Face Configuration[/bold cyan]")
        hf_dataset_name = Prompt.ask("Dataset name", default=get_random_name())
        hf_config = HuggingFaceConfig(hf_dataset_name=hf_dataset_name)
        if Confirm.ask("Configure Hugging Face options?", default=True):
            hf_config.hf_organization = Prompt.ask("Organization (use $VAR for env)", default="$HF_ORGANIZATION")
            hf_config.hf_token = Prompt.ask("HF Token (use $VAR for env)", default="$HF_TOKEN")
            hf_config.private = Confirm.ask("Make dataset private?", default=False)
            hf_config.concat_if_exist = Confirm.ask("Concatenate if dataset exists?", default=False)

            # Local dataset options
            if Confirm.ask("Configure local dataset storage?", default=False):
                hf_config.local_dataset_dir = Path(Prompt.ask("Local dataset directory", default="data/datasets"))
                hf_config.local_saving = Confirm.ask("Save dataset locally?", default=True)

        # Model configuration
        console.print("\n[bold cyan]Model Configuration[/bold cyan]")
        models = []
        add_models = Confirm.ask("Add models?", default=True)

        while add_models:
            model = create_model_config([m.model_name for m in models if m.model_name])
            models.append(model)
            add_models = Confirm.ask("\nAdd another model?", default=False)

        # Model roles
        if len(models) > 1 and Confirm.ask("\nAssign models to specific stages?", default=True):
            model_roles = configure_model_roles(models)
        else:
            model_roles = {}

        # Pipeline stages
        pipeline_config = configure_pipeline_stages()

    return YourbenchConfig(
        hf_configuration=hf_config,
        model_list=models,
        model_roles=model_roles,
        pipeline_config=pipeline_config,
    )


def save_config(config: YourbenchConfig, output_path: Path) -> None:
    """Save configuration to YAML file."""
    # Build configuration dictionary using dataclass fields
    hf_config = config.hf_configuration
    config_dict = {
        "hf_configuration": {
            "hf_dataset_name": hf_config.hf_dataset_name,
            "hf_organization": hf_config.hf_organization,
            "hf_token": hf_config.hf_token,
            "private": hf_config.private,
            "concat_if_exist": hf_config.concat_if_exist,
        },
        "model_list": [
            {
                **{
                    "model_name": model.model_name,
                    "base_url": model.base_url,
                    "api_key": model.api_key,
                    "max_concurrent_requests": model.max_concurrent_requests,
                    "encoding_name": model.encoding_name,
                    "provider": model.provider,
                },
                **({"extra_parameters": model.extra_parameters} if model.extra_parameters else {}),
            }
            for model in config.model_list
        ],
        "model_roles": config.model_roles,
        "pipeline": _build_pipeline_dict(config.pipeline_config),
    }

    # Clean None values recursively
    clean_config = _clean_none_values(config_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.dump(clean_config, default_flow_style=False, sort_keys=False, width=120))

    console.print(f"\n[green]✓[/green] Configuration saved to: {output_path}")


def _build_pipeline_dict(pipeline_config: PipelineConfig) -> dict[str, Any]:
    """Build pipeline dictionary from PipelineConfig."""
    return {
        "ingestion": {
            "run": pipeline_config.ingestion.run,
            "source_documents_dir": str(pipeline_config.ingestion.source_documents_dir),
            "output_dir": str(pipeline_config.ingestion.output_dir),
        },
        "summarization": {"run": pipeline_config.summarization.run},
        "chunking": {"run": pipeline_config.chunking.run},
        "single_shot_question_generation": {"run": pipeline_config.single_shot_question_generation.run},
        "multi_hop_question_generation": {"run": pipeline_config.multi_hop_question_generation.run},
        "cross_document_question_generation": {"run": pipeline_config.cross_document_question_generation.run},
        "question_rewriting": {"run": pipeline_config.question_rewriting.run},
        "prepare_lighteval": {"run": pipeline_config.prepare_lighteval.run},
        "citation_score_filtering": {"run": pipeline_config.citation_score_filtering.run},
    }


def _clean_none_values(obj: Any) -> Any:
    """Recursively remove None values from nested dictionaries and lists."""
    match obj:
        case dict():
            return {k: _clean_none_values(v) for k, v in obj.items() if v is not None}
        case list():
            return [_clean_none_values(item) for item in obj]
        case _:
            return obj


def load_config(config_path: Path) -> YourbenchConfig:
    """Load configuration from YAML file using configuration_engine."""
    return YourbenchConfig.from_yaml(config_path)


if __name__ == "__main__":
    # Example usage
    config = create_yourbench_config(simple=True)
    save_config(config, Path("test_config.yaml"))
