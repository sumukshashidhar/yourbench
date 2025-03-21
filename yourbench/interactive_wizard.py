# interactive_wizard.py

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import questionary
from loguru import logger
import yaml

from yourbench.config_cache import save_last_config
from yourbench.pipeline.handler import run_pipeline


@dataclass
class HFInfoBlock:
    """
    Configuration block for Hugging Face credentials and dataset details.
    """
    token: str = ""
    organization: str = ""
    private: bool = True
    global_dataset_name: str = "yourbench-wizard-example"


@dataclass
class ChunkSamplingBlock:
    """
    Configuration block for chunk sampling in question generation.
    """
    mode: str = "count"
    value: float = 5.0
    random_seed: int = 123


@dataclass
class SingleShotBlock:
    """
    Single-shot question generation stage configuration.
    """
    run: bool = False
    additional_instructions: str = "Generate questions for a curious adult"
    chunk_sampling: ChunkSamplingBlock = field(default_factory=ChunkSamplingBlock)


@dataclass
class MultiHopBlock:
    """
    Multi-hop question generation stage configuration.
    """
    run: bool = False
    additional_instructions: str = "Generate more advanced multi-hop questions"
    chunk_sampling: ChunkSamplingBlock = field(
        default_factory=lambda: ChunkSamplingBlock(
            mode="percentage", value=0.3, random_seed=42
        )
    )


@dataclass
class ChunkingParamsBlock:
    """
    Chunking parameters for the chunking stage.
    """
    l_min_tokens: int = 64
    l_max_tokens: int = 128
    tau_threshold: float = 0.8
    h_min: int = 2
    h_max: int = 5
    num_multihops_factor: int = 2


def find_latest_config_number(configs_folder: Path) -> int:
    """
    Find the highest integer label among existing created-config-XXX.yaml files
    in the given folder. Returns 0 if none found.
    """
    if not configs_folder.exists():
        return 0

    pattern = re.compile(r"^created-config-(\d{3})\.yaml$")
    highest_num = 0
    for file in configs_folder.iterdir():
        match = pattern.match(file.name)
        if match:
            num = int(match.group(1))
            highest_num = max(highest_num, num)
    return highest_num


def load_config_if_desired(configs_folder: Path) -> dict[str, Any] | None:
    """
    Ask the user if they'd like to load the most recently created config from the `configs/`
    folder. If yes, return the loaded config dictionary. Otherwise return None.
    """
    latest_num = find_latest_config_number(configs_folder)
    if latest_num == 0:
        # No existing config files
        return None

    latest_file = configs_folder / f"created-config-{latest_num:03d}.yaml"
    if not latest_file.exists():
        return None

    confirm_load = questionary.confirm(
        f"Would you like to load your last config '{latest_file.name}' and possibly modify it?",
        default=True
    ).ask()

    if not confirm_load:
        return None

    # Attempt to load
    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            loaded_cfg = yaml.safe_load(f)
        logger.info(f"Loaded existing config from {latest_file}")
        return loaded_cfg
    except Exception as e:
        logger.error(f"Failed to load {latest_file}: {e}")
        return None


def configure_hf_info(existing_block: dict[str, Any] | None) -> HFInfoBlock:
    """
    Interactively configure Hugging Face credentials/org/dataset info.
    If existing_block is provided, use those as defaults.
    """
    logger.info("Configuring Hugging Face credentials...")
    defaults = HFInfoBlock()
    if existing_block:
        defaults.token = existing_block.get("token", "")
        defaults.organization = existing_block.get("hf_organization", "")
        defaults.private = existing_block.get("private", True)
        defaults.global_dataset_name = existing_block.get("global_dataset_name", "yourbench-wizard-example")

    # Start Q&A
    default_token = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_TOKEN", "")
    if defaults.token and not default_token:
        # If the config has a token, set it as default
        default_token = defaults.token

    if default_token:
        logger.info("Found HF token in environment or existing config.")
    new_token = questionary.text(
        "Enter your Hugging Face token (leave blank to skip):",
        default=default_token
    ).ask()
    if new_token is not None:
        defaults.token = new_token.strip()

    default_org = defaults.organization or os.environ.get("HF_USERNAME", "") \
        or os.environ.get("HUGGINGFACE_USERNAME", "") or os.environ.get("HF_ORG", "")

    new_org = questionary.text(
        "Enter your Hugging Face username/organization:",
        default=default_org
    ).ask()
    if new_org is not None:
        defaults.organization = new_org.strip()

    defaults.private = questionary.confirm(
        "Do you want your dataset to be private?",
        default=defaults.private
    ).ask()

    ds_default = defaults.global_dataset_name
    ds_name = questionary.text(
        "Enter the Hugging Face dataset name (e.g. username/my-cool-dataset):",
        default=ds_default
    ).ask()
    if ds_name:
        defaults.global_dataset_name = ds_name.strip()

    return defaults


def configure_chunking_params(existing_block: dict[str, Any] | None) -> ChunkingParamsBlock:
    """
    Interactively configure chunking parameters for the chunking stage.
    """
    logger.info("Configuring chunking parameters...")
    defaults = ChunkingParamsBlock()
    if existing_block:
        defaults.l_min_tokens = existing_block.get("l_min_tokens", defaults.l_min_tokens)
        defaults.l_max_tokens = existing_block.get("l_max_tokens", defaults.l_max_tokens)
        defaults.tau_threshold = existing_block.get("tau_threshold", defaults.tau_threshold)
        defaults.h_min = existing_block.get("h_min", defaults.h_min)
        defaults.h_max = existing_block.get("h_max", defaults.h_max)
        defaults.num_multihops_factor = existing_block.get("num_multihops_factor", defaults.num_multihops_factor)

    val_str = questionary.text(
        "Min tokens per chunk?",
        default=str(defaults.l_min_tokens)
    ).ask()
    if val_str:
        defaults.l_min_tokens = int(val_str)

    val_str = questionary.text(
        "Max tokens per chunk?",
        default=str(defaults.l_max_tokens)
    ).ask()
    if val_str:
        defaults.l_max_tokens = int(val_str)

    val_str = questionary.text(
        "Tau threshold?",
        default=str(defaults.tau_threshold)
    ).ask()
    if val_str:
        defaults.tau_threshold = float(val_str)

    val_str = questionary.text(
        "h_min (minimum chunk combo)?",
        default=str(defaults.h_min)
    ).ask()
    if val_str:
        defaults.h_min = int(val_str)

    val_str = questionary.text(
        "h_max (maximum chunk combo)?",
        default=str(defaults.h_max)
    ).ask()
    if val_str:
        defaults.h_max = int(val_str)

    val_str = questionary.text(
        "num_multihops_factor?",
        default=str(defaults.num_multihops_factor)
    ).ask()
    if val_str:
        defaults.num_multihops_factor = int(val_str)

    return defaults


def configure_single_shot_block(existing_block: dict[str, Any] | None) -> SingleShotBlock:
    """
    Interactively configure single-shot question generation stage.
    """
    logger.info("Configuring single-shot question generation...")
    defaults = SingleShotBlock(run=True)
    if existing_block:
        defaults.additional_instructions = existing_block.get("additional_instructions", defaults.additional_instructions)
        chunk_sampling = existing_block.get("chunk_sampling", {})
        defaults.chunk_sampling.mode = chunk_sampling.get("mode", defaults.chunk_sampling.mode)
        defaults.chunk_sampling.value = chunk_sampling.get("value", defaults.chunk_sampling.value)
        defaults.chunk_sampling.random_seed = chunk_sampling.get("random_seed", defaults.chunk_sampling.random_seed)

    # Additional instructions
    new_instructions = questionary.text(
        "Additional instructions for single-shot QG?",
        default=defaults.additional_instructions
    ).ask()
    if new_instructions:
        defaults.additional_instructions = new_instructions.strip()

    # chunk sampling
    if questionary.confirm(
        "Would you like to configure chunk sampling? (Otherwise use defaults)",
        default=True
    ).ask():
        new_mode = questionary.select(
            "Sampling mode?",
            choices=["all", "count", "percentage"],
            default=defaults.chunk_sampling.mode
        ).ask()
        if new_mode:
            defaults.chunk_sampling.mode = new_mode.strip()

        val_str = questionary.text(
            f"Sampling value? (current={defaults.chunk_sampling.value})"
        ).ask()
        if val_str:
            defaults.chunk_sampling.value = float(val_str)

        seed_str = questionary.text(
            f"Random seed? (current={defaults.chunk_sampling.random_seed})"
        ).ask()
        if seed_str:
            defaults.chunk_sampling.random_seed = int(seed_str)

    return defaults


def configure_multi_hop_block(existing_block: dict[str, Any] | None) -> MultiHopBlock:
    """
    Interactively configure multi-hop question generation stage.
    """
    logger.info("Configuring multi-hop question generation...")
    defaults = MultiHopBlock(run=True)
    if existing_block:
        defaults.additional_instructions = existing_block.get("additional_instructions", defaults.additional_instructions)
        chunk_sampling = existing_block.get("chunk_sampling", {})
        defaults.chunk_sampling.mode = chunk_sampling.get("mode", defaults.chunk_sampling.mode)
        defaults.chunk_sampling.value = chunk_sampling.get("value", defaults.chunk_sampling.value)
        defaults.chunk_sampling.random_seed = chunk_sampling.get("random_seed", defaults.chunk_sampling.random_seed)

    # Additional instructions
    new_instructions = questionary.text(
        "Additional instructions for multi-hop QG?",
        default=defaults.additional_instructions
    ).ask()
    if new_instructions:
        defaults.additional_instructions = new_instructions.strip()

    # chunk sampling
    if questionary.confirm(
        "Configure chunk sampling for multi-hop QG?",
        default=True
    ).ask():
        new_mode = questionary.select(
            "Sampling mode?",
            choices=["all", "count", "percentage"],
            default=defaults.chunk_sampling.mode
        ).ask()
        if new_mode:
            defaults.chunk_sampling.mode = new_mode.strip()

        val_str = questionary.text(
            f"Sampling value? (current={defaults.chunk_sampling.value})"
        ).ask()
        if val_str:
            defaults.chunk_sampling.value = float(val_str)

        seed_str = questionary.text(
            f"Random seed? (current={defaults.chunk_sampling.random_seed})"
        ).ask()
        if seed_str:
            defaults.chunk_sampling.random_seed = int(seed_str)

    return defaults


def run_interactive_wizard() -> dict[str, Any]:
    """
    High-level function that runs the entire interactive wizard, 
    storing final config into configs/created-config-XXX.yaml.
    """
    questionary.print("\nWelcome to the YourBench Interactive Wizard!", style="bold")
    questionary.print("Follow the prompts to generate or update a pipeline config.\n")

    # Attempt to load an existing config if user wants
    configs_folder = Path("configs")
    configs_folder.mkdir(parents=True, exist_ok=True)

    existing_cfg: dict[str, Any] | None = load_config_if_desired(configs_folder)

    # 1) Hugging Face Info
    hf_dict = existing_cfg.get("hf_configuration") if existing_cfg else None
    hf_block = HFInfoBlock()
    if questionary.confirm("Would you like to configure Hugging Face block?").ask():
        hf_block = configure_hf_info(hf_dict)

    # 2) Pipeline stage selection
    possible_stages: list[str] = [
        "ingestion",
        "upload_ingest_to_hub",
        "summarization",
        "chunking",
        "single_shot_question_generation",
        "multi_hop_question_generation",
        "deduplicate_single_shot_questions",
        "deduplicate_multi_hop_questions",
        "lighteval",
    ]
    questionary.print("\n=== Pipeline Stages ===\n", style="bold")
    if existing_cfg and "pipeline" in existing_cfg:
        already_enabled = [k for k, v in existing_cfg["pipeline"].items() if v.get("run")]
        chosen_stages = questionary.checkbox(
            "Select pipeline stages to enable:",
            choices=possible_stages,
            default=already_enabled or [],
        ).ask() or []
    else:
        chosen_stages = questionary.checkbox(
            "Select pipeline stages to enable:",
            choices=possible_stages
        ).ask() or []

    pipeline_cfg: dict[str, Any] = {}

    # ingestion
    old_ingestion = (existing_cfg["pipeline"].get("ingestion") if existing_cfg and "ingestion" in chosen_stages else None)
    if "ingestion" in chosen_stages:
        if questionary.confirm("Configure ingestion block now?", default=True).ask():
            src_dir = questionary.text(
                "Local directory for raw source documents?",
                default=old_ingestion.get("source_documents_dir", "data/example/raw") if old_ingestion else "data/example/raw"
            ).ask() or "data/example/raw"

            out_dir = questionary.text(
                "Directory to store ingested .md files?",
                default=old_ingestion.get("output_dir", "data/example/processed") if old_ingestion else "data/example/processed"
            ).ask() or "data/example/processed"

            pipeline_cfg["ingestion"] = {
                "run": True,
                "source_documents_dir": src_dir.strip(),
                "output_dir": out_dir.strip()
            }
        else:
            pipeline_cfg["ingestion"] = {"run": True}
    else:
        pipeline_cfg["ingestion"] = {"run": False}

    # upload
    old_upload = (existing_cfg["pipeline"].get("upload_ingest_to_hub") if existing_cfg and "upload_ingest_to_hub" in chosen_stages else None)
    if "upload_ingest_to_hub" in chosen_stages:
        if questionary.confirm("Configure upload_ingest_to_hub now?", default=True).ask():
            up_dir = questionary.text(
                "Which directory to upload to HF Hub?",
                default=old_upload.get("source_documents_dir", "data/example/processed") if old_upload else "data/example/processed"
            ).ask() or "data/example/processed"
            pipeline_cfg["upload_ingest_to_hub"] = {
                "run": True,
                "source_documents_dir": up_dir.strip(),
                "output_subset": "ingested_documents"
            }
        else:
            pipeline_cfg["upload_ingest_to_hub"] = {"run": True}
    else:
        pipeline_cfg["upload_ingest_to_hub"] = {"run": False}

    # summarization
    pipeline_cfg["summarization"] = {
        "run": ("summarization" in chosen_stages)
    }

    # chunking
    old_chunking = (existing_cfg["pipeline"].get("chunking") if existing_cfg and "chunking" in chosen_stages else None)
    if "chunking" in chosen_stages:
        if questionary.confirm("Configure chunking stage now?", default=True).ask():
            chunking_block = configure_chunking_params(old_chunking.get("chunking_configuration") if old_chunking else None)
            pipeline_cfg["chunking"] = {
                "run": True,
                "chunking_configuration": {
                    "l_min_tokens": chunking_block.l_min_tokens,
                    "l_max_tokens": chunking_block.l_max_tokens,
                    "tau_threshold": chunking_block.tau_threshold,
                    "h_min": chunking_block.h_min,
                    "h_max": chunking_block.h_max,
                    "num_multihops_factor": chunking_block.num_multihops_factor
                }
            }
        else:
            pipeline_cfg["chunking"] = {"run": True}
    else:
        pipeline_cfg["chunking"] = {"run": False}

    # single_shot
    old_single_shot = None
    if existing_cfg and "pipeline" in existing_cfg:
        old_single_shot = existing_cfg["pipeline"].get("single_shot_question_generation")

    if "single_shot_question_generation" in chosen_stages:
        if questionary.confirm("Configure single-shot question generation now?", default=True).ask():
            single_shot = configure_single_shot_block(old_single_shot)
            pipeline_cfg["single_shot_question_generation"] = {
                "run": True,
                "additional_instructions": single_shot.additional_instructions,
                "chunk_sampling": {
                    "mode": single_shot.chunk_sampling.mode,
                    "value": single_shot.chunk_sampling.value,
                    "random_seed": single_shot.chunk_sampling.random_seed
                }
            }
        else:
            pipeline_cfg["single_shot_question_generation"] = {"run": True}
    else:
        pipeline_cfg["single_shot_question_generation"] = {"run": False}

    # multi_hop
    old_multi_hop = None
    if existing_cfg and "pipeline" in existing_cfg:
        old_multi_hop = existing_cfg["pipeline"].get("multi_hop_question_generation")

    if "multi_hop_question_generation" in chosen_stages:
        if questionary.confirm("Configure multi-hop question generation now?", default=True).ask():
            multi_hop = configure_multi_hop_block(old_multi_hop)
            pipeline_cfg["multi_hop_question_generation"] = {
                "run": True,
                "additional_instructions": multi_hop.additional_instructions,
                "chunk_sampling": {
                    "mode": multi_hop.chunk_sampling.mode,
                    "value": multi_hop.chunk_sampling.value,
                    "random_seed": multi_hop.chunk_sampling.random_seed
                }
            }
        else:
            pipeline_cfg["multi_hop_question_generation"] = {"run": True}
    else:
        pipeline_cfg["multi_hop_question_generation"] = {"run": False}

    # deduplicate
    pipeline_cfg["deduplicate_single_shot_questions"] = {
        "run": "deduplicate_single_shot_questions" in chosen_stages
    }
    pipeline_cfg["deduplicate_multi_hop_questions"] = {
        "run": "deduplicate_multi_hop_questions" in chosen_stages
    }
    pipeline_cfg["lighteval"] = {"run": "lighteval" in chosen_stages}

    # 3) Model Setup
    old_model_list = existing_cfg.get("model_list") if existing_cfg else []
    old_roles = existing_cfg.get("model_roles") if existing_cfg else {}

    questionary.print("\n=== Model Configuration ===\n", style="bold")
    model_list: list[dict[str, Any]] = []
    model_roles: dict[str, list[str]] = {stg: [] for stg in pipeline_cfg.keys()}

    confirm_models = questionary.confirm(
        "Would you like to configure any models? (Replaces any existing model_list)"
    ).ask()
    if confirm_models:
        count_str = questionary.text("How many models do you want to configure?", default="1").ask() or "1"
        try:
            count = int(count_str)
        except ValueError:
            count = 1

        for i in range(count):
            questionary.print(f"\nConfiguring Model #{i+1}", style="bold")
            model_name = questionary.text(
                "Enter a model identifier (e.g. openai-gpt-3.5-turbo):",
                default=f"my-model-{i+1}"
            ).ask() or f"my-model-{i+1}"

            inference_backend = questionary.select(
                "Which inference backend?",
                choices=["litellm", "hf_hub"],
                default="litellm"
            ).ask() or "litellm"

            request_style = ""
            provider = None
            if inference_backend == "litellm":
                request_style = questionary.select(
                    "Select request_style for litellm:",
                    choices=["openai", "anthropic", "cohere", "google", "azure", "replicate"],
                    default="openai"
                ).ask() or "openai"

            elif inference_backend == "hf_hub":
                provider = questionary.text(
                    "Enter HF Hub provider (optional):",
                    default=""
                ).ask() or None

            base_url = questionary.text(
                "Enter model base_url (if needed). Leave blank if none:",
                default=""
            ).ask() or ""

            default_api_key = ""
            # environment-based
            if inference_backend == "litellm" and request_style == "openai":
                default_api_key = os.environ.get("OPENAI_API_KEY", "")
            elif inference_backend == "hf_hub":
                default_api_key = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_TOKEN", "")
            # model-specific
            model_key_var = f"{model_name.upper().replace('-', '_')}_API_KEY"
            if os.environ.get(model_key_var):
                default_api_key = os.environ[model_key_var]

            api_key = questionary.text(
                f"Enter API key for {model_name} (leave blank if none):",
                default=default_api_key
            ).ask() or ""

            api_version = questionary.text(
                "Enter API version (if needed, e.g. Azure). Blank if none:",
                default=""
            ).ask() or None

            concurrency_str = questionary.text(
                "Max concurrent requests?",
                default="8"
            ).ask() or "8"
            try:
                concurrency_val = int(concurrency_str)
            except ValueError:
                concurrency_val = 8

            # Pipeline roles
            assigned_stages = questionary.checkbox(
                f"Which pipeline stages for {model_name}?",
                choices=list(pipeline_cfg.keys())
            ).ask() or []

            single_model = {
                "model_name": model_name.strip(),
                "request_style": request_style.strip() if request_style else None,
                "base_url": base_url.strip() if base_url else None,
                "api_key": api_key.strip() if api_key else None,
                "max_concurrent_requests": concurrency_val,
                "inference_backend": inference_backend.strip(),
                "provider": provider.strip() if provider else None,
                "api_version": api_version.strip() if api_version else None
            }
            model_list.append(single_model)

            for stg in assigned_stages:
                if stg in model_roles:
                    model_roles[stg].append(model_name.strip())
    else:
        # Possibly re-use old model_list if it existed
        if old_model_list:
            model_list = old_model_list
        # model_roles defaults or from old
        if old_roles:
            for stg, names in old_roles.items():
                if stg in model_roles:
                    model_roles[stg] = names

    # Final config
    config_dict = {
        "hf_configuration": {
            "token": hf_block.token,
            "hf_organization": hf_block.organization,
            "private": hf_block.private,
            "global_dataset_name": hf_block.global_dataset_name
        },
        "pipeline": pipeline_cfg,
        "model_list": model_list,
        "model_roles": model_roles
    }

    # Save to configs/ with next integer label
    latest_num = find_latest_config_number(configs_folder)
    next_num = latest_num + 1
    new_config_path = configs_folder / f"created-config-{next_num:03d}.yaml"
    with open(new_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)

    logger.info(f"Wizard config saved to: {new_config_path}")
    save_last_config(str(new_config_path))

    # Optionally run pipeline
    if questionary.confirm("Run pipeline now with newly created (or updated) config?").ask():
        run_pipeline(str(new_config_path), debug=False, plot_stage_timing=False)

    return config_dict
