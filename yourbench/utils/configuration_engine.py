"""
Module handles everything related to the configuration of the pipeline.
"""

import os
from typing import Any
from pathlib import Path
from dataclasses import field, fields, dataclass

import yaml
from loguru import logger
from randomname import get_name as get_random_name

from huggingface_hub import whoami


def _expand_env(value: Any) -> Any:
    """
    Replace leading '$VARNAME' with its environment value.
    Special case: if $HF_ORGANIZATION is missing we try HF_TOKEN + whoami().
    """
    if not (isinstance(value, str) and value.startswith("$")):
        return value

    var = value[1:]
    if env := os.getenv(var):
        return env

    # == SPECIAL CASES ==
    if var == "HF_ORGANIZATION":
        token = os.getenv("HF_TOKEN")
        if token:
            try:
                return whoami(token)["name"]
            except Exception:
                logger.warning("Failed to get organization name from HF_TOKEN. Push to hub will fail.")
                pass  # fall through and return literal
    return value


def _expand_dataclass(obj: Any) -> None:
    """In-place $ENV expansion for every str field of a dataclass."""
    for f in fields(obj):
        setattr(obj, f.name, _expand_env(getattr(obj, f.name)))


@dataclass
class HuggingFaceConfig:
    """Configuration for the Hugging Face dataset."""

    hf_dataset_name: str = get_random_name()
    hf_organization: str = "$HF_ORGANIZATION"
    hf_token: str = "$HF_TOKEN"
    private: bool = False
    concat_if_exist: bool = False
    local_dataset_dir: Path | None = Path("data/saved_dataset")
    local_saving: bool = True
    upload_card: bool = True

    def __post_init__(self) -> None:
        _expand_dataclass(self)


@dataclass
class ModelConfig:
    """Configuration for a model."""

    model_name: str | None = None
    base_url: str | None = None
    api_key: str | None = "$HF_TOKEN"
    max_concurrent_requests: int = 32
    encoding_name: str = "cl100k_base"

    # You can find the list of available providers here: https://huggingface.co/docs/huggingface_hub/guides/inference#supported-providers-and-tasks
    # huggingface specific
    provider: str | None = None
    bill_to: str | None = None

    def __post_init__(self) -> None:
        _expand_dataclass(self)

        # if base_url is not set, and provider is not set, default to "auto"
        if not self.base_url and not self.provider:
            self.provider = "auto"


@dataclass
class IngestionConfig:
    """Configuration for the ingestion stage"""

    run: bool = False
    source_documents_dir: Path | None = Path("example/data/raw/simple_example")
    output_dir: Path | None = Path("data/processed/simple_example")
    upload_to_hub: bool = True
    llm_ingestion: bool = False
    pdf_dpi: int = 300
    pdf_llm_prompt: str | Path = Path("yourbench/prompts/ingestion/pdf_llm_prompt.md")
    supported_file_extensions: list[str] = field(
        default_factory=lambda: [
            ".md",
            ".txt",
            ".html",
            ".htm",
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".xlsx",
            ".xls",
            ".rtf",
            ".odt",
        ]
    )

    def __post_init__(self) -> None:
        # convert string directories to Path objects
        if self.source_documents_dir:
            self.source_documents_dir = Path(self.source_documents_dir)
        if self.output_dir:
            self.output_dir = Path(self.output_dir)

        prompt_path = Path(self.pdf_llm_prompt)
        if prompt_path.is_file():
            self.pdf_llm_prompt = prompt_path.read_text(encoding="utf-8").strip()

        if not self.source_documents_dir or not self.output_dir:
            logger.error("Missing source or output director. Creating default directories.")
            raise ValueError("Missing source or output directory")


@dataclass
class SummarizationConfig:
    """Configuration for the summarization stage"""

    run: bool = False
    max_tokens: int = 32768
    token_overlap: int = 512
    encoding_name: str = "cl100k_base"
    summarization_user_prompt: str | Path = Path("yourbench/prompts/summarization/summarization_user_prompt.md")
    combine_summaries_user_prompt: str | Path = Path(
        "yourbench/prompts/summarization/combine_summaries_user_prompt.md"
    )

    def __post_init__(self) -> None:
        # Load prompt files if they exist
        summarization_prompt_path = Path(self.summarization_user_prompt)
        if summarization_prompt_path.is_file():
            self.summarization_user_prompt = summarization_prompt_path.read_text(encoding="utf-8").strip()

        combine_prompt_path = Path(self.combine_summaries_user_prompt)
        if combine_prompt_path.is_file():
            self.combine_summaries_user_prompt = combine_prompt_path.read_text(encoding="utf-8").strip()


@dataclass
class ChunkingConfig:
    """Configuration for the chunking stage"""

    run: bool = False
    l_max_tokens: int = 8192
    token_overlap: int = 512
    encoding_name: str = "cl100k_base"
    h_min: int = 2
    h_max: int = 5
    num_multihops_factor: int = 1


@dataclass
class QuestionGenerationConfig:
    """Configuration for the question generation stage"""

    run: bool = False


@dataclass
class SingleShotQuestionGenerationConfig:
    """Configuration for the single shot question generation stage"""

    run: bool = False
    question_mode: str = "open-ended"  # "open-ended" or "multi-choice"
    single_shot_system_prompt: str | Path = Path("yourbench/prompts/question_generation/single_shot_system_prompt.md")
    single_shot_system_prompt_multi: str | Path = Path(
        "yourbench/prompts/question_generation/single_shot_system_prompt_multi.md"
    )
    single_shot_user_prompt: str | Path = Path("yourbench/prompts/question_generation/single_shot_user_prompt.md")
    additional_instructions: str = ""

    def __post_init__(self) -> None:
        # Load prompt files if they exist
        single_shot_system_prompt_path = Path(self.single_shot_system_prompt)
        if single_shot_system_prompt_path.is_file():
            self.single_shot_system_prompt = single_shot_system_prompt_path.read_text(encoding="utf-8").strip()

        single_shot_system_prompt_multi_path = Path(self.single_shot_system_prompt_multi)
        if single_shot_system_prompt_multi_path.is_file():
            self.single_shot_system_prompt_multi = single_shot_system_prompt_multi_path.read_text(
                encoding="utf-8"
            ).strip()

        single_shot_user_prompt_path = Path(self.single_shot_user_prompt)
        if single_shot_user_prompt_path.is_file():
            self.single_shot_user_prompt = single_shot_user_prompt_path.read_text(encoding="utf-8").strip()


@dataclass
class MultiHopQuestionGenerationConfig:
    """Configuration for the multi hop question generation stage"""

    run: bool = False
    question_mode: str = "open-ended"  # "open-ended" or "multi-choice"
    multi_hop_system_prompt: str | Path = Path("yourbench/prompts/question_generation/multi_hop_system_prompt.md")
    multi_hop_system_prompt_multi: str | Path = Path(
        "yourbench/prompts/question_generation/multi_hop_system_prompt_multi.md"
    )
    multi_hop_user_prompt: str | Path = Path("yourbench/prompts/question_generation/multi_hop_user_prompt.md")
    additional_instructions: str = ""

    def __post_init__(self) -> None:
        # Load prompt files if they exist
        multi_hop_system_prompt_path = Path(self.multi_hop_system_prompt)
        if multi_hop_system_prompt_path.is_file():
            self.multi_hop_system_prompt = multi_hop_system_prompt_path.read_text(encoding="utf-8").strip()

        multi_hop_system_prompt_multi_path = Path(self.multi_hop_system_prompt_multi)
        if multi_hop_system_prompt_multi_path.is_file():
            self.multi_hop_system_prompt_multi = multi_hop_system_prompt_multi_path.read_text(encoding="utf-8").strip()

        multi_hop_user_prompt_path = Path(self.multi_hop_user_prompt)
        if multi_hop_user_prompt_path.is_file():
            self.multi_hop_user_prompt = multi_hop_user_prompt_path.read_text(encoding="utf-8").strip()


@dataclass
class CrossDocumentQuestionGenerationConfig:
    """Configuration for the cross-document question generation stage"""

    run: bool = False
    question_mode: str = "open-ended"  # "open-ended" or "multi-choice"
    multi_hop_system_prompt: str | Path = Path("yourbench/prompts/question_generation/multi_hop_system_prompt.md")
    multi_hop_system_prompt_multi: str | Path = Path(
        "yourbench/prompts/question_generation/multi_hop_system_prompt_multi.md"
    )
    multi_hop_user_prompt: str | Path = Path("yourbench/prompts/question_generation/multi_hop_user_prompt.md")
    additional_instructions: str = ""
    max_combinations: int = 100
    chunks_per_document: int = 1
    num_docs_per_combination: list[int] = field(default_factory=lambda: [2, 5])
    random_seed: int = 42

    def __post_init__(self) -> None:
        # Load prompt files if they exist
        multi_hop_system_prompt_path = Path(self.multi_hop_system_prompt)
        if multi_hop_system_prompt_path.is_file():
            self.multi_hop_system_prompt = multi_hop_system_prompt_path.read_text(encoding="utf-8").strip()

        multi_hop_system_prompt_multi_path = Path(self.multi_hop_system_prompt_multi)
        if multi_hop_system_prompt_multi_path.is_file():
            self.multi_hop_system_prompt_multi = multi_hop_system_prompt_multi_path.read_text(encoding="utf-8").strip()

        multi_hop_user_prompt_path = Path(self.multi_hop_user_prompt)
        if multi_hop_user_prompt_path.is_file():
            self.multi_hop_user_prompt = multi_hop_user_prompt_path.read_text(encoding="utf-8").strip()


@dataclass
class QuestionRewritingConfig:
    """Configuration for the question rewriting stage"""

    run: bool = False
    question_rewriting_system_prompt: str | Path = Path(
        "yourbench/prompts/question_rewriting/question_rewriting_system_prompt.md"
    )
    question_rewriting_user_prompt: str | Path = Path(
        "yourbench/prompts/question_rewriting/question_rewriting_user_prompt.md"
    )
    additional_instructions: str = (
        "Rewrite the question to sound more natural and conversational while preserving the exact meaning."
    )

    def __post_init__(self) -> None:
        # Load prompt files if they exist
        system_prompt_path = Path(self.question_rewriting_system_prompt)
        if system_prompt_path.is_file():
            self.question_rewriting_system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()

        user_prompt_path = Path(self.question_rewriting_user_prompt)
        if user_prompt_path.is_file():
            self.question_rewriting_user_prompt = user_prompt_path.read_text(encoding="utf-8").strip()


@dataclass
class LightevalConfig:
    """Configuration for the lighteval stage"""

    run: bool = False


@dataclass
class CitationScoreFilteringConfig:
    """Configuration for the citation score filtering stage"""

    run: bool = False


@dataclass
class PipelineConfig:
    """Configuration for the pipeline"""

    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    question_generation: QuestionGenerationConfig = field(default_factory=QuestionGenerationConfig)
    single_shot_question_generation: SingleShotQuestionGenerationConfig = field(
        default_factory=SingleShotQuestionGenerationConfig
    )
    multi_hop_question_generation: MultiHopQuestionGenerationConfig = field(
        default_factory=MultiHopQuestionGenerationConfig
    )
    cross_document_question_generation: CrossDocumentQuestionGenerationConfig = field(
        default_factory=CrossDocumentQuestionGenerationConfig
    )
    question_rewriting: QuestionRewritingConfig = field(default_factory=QuestionRewritingConfig)
    lighteval: LightevalConfig = field(default_factory=LightevalConfig)
    prepare_lighteval: LightevalConfig = field(default_factory=LightevalConfig)
    citation_score_filtering: CitationScoreFilteringConfig = field(default_factory=CitationScoreFilteringConfig)


@dataclass
class YourbenchConfig:
    """The main configuration class for the YourBench pipeline."""

    hf_configuration: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    model_list: list[ModelConfig] = field(default_factory=list)
    model_roles: dict[str, list[str]] = field(default_factory=dict)
    debug: bool = False

    def __post_init__(self) -> None:
        """Assign default model roles for each pipeline stage if not specified."""
        if not self.model_list:
            return

        # Get the first model name as default
        default_model = self.model_list[0].model_name
        if not default_model:
            return

        # All pipeline stages that can use models
        pipeline_stages = [
            "ingestion",
            "summarization",
            "chunking",
            "question_generation",
            "single_shot_question_generation",
            "multi_hop_question_generation",
            "cross_document_question_generation",
            "question_rewriting",
            "prepare_lighteval",
            "citation_score_filtering",
        ]

        # Assign default model to stages that don't have model roles defined
        for stage in pipeline_stages:
            if stage not in self.model_roles:
                self.model_roles[stage] = [default_model]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "YourbenchConfig":
        """
        Load YAML → dict → dataclass, with env-var expansion
        confined to HuggingFaceConfig.__post_init__.
        """
        with open(Path(path), "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        hf_kwargs = data.get("hf_configuration", {})

        # Handle both 'models' and 'model_list' keys for backward compatibility
        model_list = data.get("model_list", data.get("models", []))
        model_roles = data.get("model_roles", {})

        # Handle pipeline configuration with proper nested dataclass instantiation
        pipeline_data = data.get("pipeline", {})
        pipeline_kwargs = {}

        # Map stage names to their corresponding config classes
        stage_config_classes = {
            "ingestion": IngestionConfig,
            "summarization": SummarizationConfig,
            "chunking": ChunkingConfig,
            "question_generation": QuestionGenerationConfig,
            "single_shot_question_generation": SingleShotQuestionGenerationConfig,
            "multi_hop_question_generation": MultiHopQuestionGenerationConfig,
            "cross_document_question_generation": CrossDocumentQuestionGenerationConfig,
            "question_rewriting": QuestionRewritingConfig,
            "lighteval": LightevalConfig,
            "prepare_lighteval": LightevalConfig,
            "citation_score_filtering": CitationScoreFilteringConfig,
        }

        # Convert each stage configuration dict to its dataclass instance
        for stage_name, config_data in pipeline_data.items():
            if stage_name in stage_config_classes:
                config_class = stage_config_classes[stage_name]

                if config_data is None:
                    config_data = {}

                # If a stage is present in the config, assume it should run unless `run: false` is explicit.
                config_data.setdefault("run", True)

                pipeline_kwargs[stage_name] = config_class(**config_data)
            else:
                logger.warning(f"Unknown pipeline stage: {stage_name}")

        return cls(
            hf_configuration=HuggingFaceConfig(**hf_kwargs),
            model_list=[ModelConfig(**m) for m in model_list],
            model_roles=model_roles,
            pipeline_config=PipelineConfig(**pipeline_kwargs),
        )


if __name__ == "__main__":
    config = YourbenchConfig.from_yaml("example/configs/simple_example.yaml")
    print(config)
