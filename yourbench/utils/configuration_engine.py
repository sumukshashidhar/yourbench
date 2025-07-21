"""
Module handles everything related to the configuration of the pipeline.
"""

import os
from typing import Union
from pathlib import Path

import yaml
from loguru import logger
from pydantic import (
    Field,
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)
from randomname import get_name as get_random_name

from huggingface_hub import whoami


def _expand_env(value: str) -> str:
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


class HuggingFaceConfig(BaseModel):
    """Configuration for the Hugging Face dataset."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    hf_dataset_name: str = Field(default_factory=get_random_name)
    hf_organization: str = "$HF_ORGANIZATION"
    hf_token: str = "$HF_TOKEN"
    private: bool = False
    concat_if_exist: bool = False
    local_dataset_dir: Path | None = Path("data/saved_dataset")
    local_saving: bool = True
    upload_card: bool = True

    @field_validator("hf_organization", "hf_token")
    @classmethod
    def expand_env_vars(cls, v: str) -> str:
        return _expand_env(v)

    @field_validator("local_dataset_dir")
    @classmethod
    def validate_path(cls, v: Union[str, Path, None]) -> Path | None:
        if v is None:
            return None
        return Path(v)


class ModelConfig(BaseModel):
    """Configuration for a model."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    model_name: str | None = None
    base_url: str | None = None
    api_key: str | None = "$HF_TOKEN"
    max_concurrent_requests: int = Field(default=32, ge=1, le=100)
    encoding_name: str = "cl100k_base"
    provider: str | None = None
    bill_to: str | None = None

    @field_validator("api_key")
    @classmethod
    def expand_env_vars(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _expand_env(v)

    @model_validator(mode="after")
    def set_default_provider(self):
        if not self.base_url and not self.provider:
            self.provider = "auto"
        return self


class IngestionConfig(BaseModel):
    """Configuration for the ingestion stage."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    run: bool = False
    source_documents_dir: Path = Path("example/data/raw/simple_example")
    output_dir: Path = Path("data/processed/simple_example")
    upload_to_hub: bool = True
    llm_ingestion: bool = False
    pdf_dpi: int = Field(default=300, ge=72, le=600)
    pdf_llm_prompt: str = Field(default="")
    supported_file_extensions: list[str] = Field(
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

    @field_validator("source_documents_dir", "output_dir")
    @classmethod
    def validate_path(cls, v: Union[str, Path]) -> Path:
        return Path(v)

    @model_validator(mode="after")
    def load_prompt_and_validate_dirs(self):
        # Load prompt if it's a file path
        prompt_path = Path("yourbench/prompts/ingestion/pdf_llm_prompt.md")
        if prompt_path.exists():
            self.pdf_llm_prompt = prompt_path.read_text(encoding="utf-8").strip()

        # Validate directories exist or can be created
        if not self.source_documents_dir.exists():
            logger.warning(f"Source directory does not exist: {self.source_documents_dir}")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        return self


class SummarizationConfig(BaseModel):
    """Configuration for the summarization stage."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    run: bool = False
    max_tokens: int = Field(default=32768, ge=1024, le=100000)
    token_overlap: int = Field(default=512, ge=0)
    encoding_name: str = "cl100k_base"
    summarization_user_prompt: str = Field(default="")
    combine_summaries_user_prompt: str = Field(default="")

    @model_validator(mode="after")
    def load_prompts(self):
        # Load summarization prompt
        sum_path = Path("yourbench/prompts/summarization/summarization_user_prompt.md")
        if sum_path.exists():
            self.summarization_user_prompt = sum_path.read_text(encoding="utf-8").strip()

        # Load combine summaries prompt
        combine_path = Path("yourbench/prompts/summarization/combine_summaries_user_prompt.md")
        if combine_path.exists():
            self.combine_summaries_user_prompt = combine_path.read_text(encoding="utf-8").strip()

        return self


class ChunkingConfig(BaseModel):
    """Configuration for the chunking stage."""

    model_config = ConfigDict(validate_assignment=True)

    run: bool = False
    l_max_tokens: int = Field(default=8192, ge=256, le=50000)
    token_overlap: int = Field(default=512, ge=0)
    encoding_name: str = "cl100k_base"
    h_min: int = Field(default=2, ge=1)
    h_max: int = Field(default=5, ge=1)
    num_multihops_factor: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_hop_ranges(self):
        if self.h_min > self.h_max:
            raise ValueError(f"h_min ({self.h_min}) cannot be greater than h_max ({self.h_max})")
        return self


class QuestionGenerationConfig(BaseModel):
    """Base configuration for question generation stages."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    run: bool = False
    question_mode: str = Field(default="open-ended", pattern="^(open-ended|multi-choice)$")
    additional_instructions: str = ""


class SingleShotQuestionGenerationConfig(QuestionGenerationConfig):
    """Configuration for single shot question generation."""

    single_shot_system_prompt: str = Field(default="")
    single_shot_system_prompt_multi: str = Field(default="")
    single_shot_user_prompt: str = Field(default="")

    @model_validator(mode="after")
    def load_prompts(self):
        # Load prompts from files if they exist
        prompts_map = {
            "single_shot_system_prompt": "yourbench/prompts/question_generation/single_shot_system_prompt.md",
            "single_shot_system_prompt_multi": "yourbench/prompts/question_generation/single_shot_system_prompt_multi.md",
            "single_shot_user_prompt": "yourbench/prompts/question_generation/single_shot_user_prompt.md",
        }

        for attr, path_str in prompts_map.items():
            path = Path(path_str)
            if path.exists():
                setattr(self, attr, path.read_text(encoding="utf-8").strip())

        return self


class MultiHopQuestionGenerationConfig(QuestionGenerationConfig):
    """Configuration for multi-hop question generation."""

    multi_hop_system_prompt: str = Field(default="")
    multi_hop_system_prompt_multi: str = Field(default="")
    multi_hop_user_prompt: str = Field(default="")

    @model_validator(mode="after")
    def load_prompts(self):
        # Load prompts from files if they exist
        prompts_map = {
            "multi_hop_system_prompt": "yourbench/prompts/question_generation/multi_hop_system_prompt.md",
            "multi_hop_system_prompt_multi": "yourbench/prompts/question_generation/multi_hop_system_prompt_multi.md",
            "multi_hop_user_prompt": "yourbench/prompts/question_generation/multi_hop_user_prompt.md",
        }

        for attr, path_str in prompts_map.items():
            path = Path(path_str)
            if path.exists():
                setattr(self, attr, path.read_text(encoding="utf-8").strip())

        return self


class CrossDocumentQuestionGenerationConfig(QuestionGenerationConfig):
    """Configuration for cross-document question generation."""

    multi_hop_system_prompt: str = Field(default="")
    multi_hop_system_prompt_multi: str = Field(default="")
    multi_hop_user_prompt: str = Field(default="")
    max_combinations: int = Field(default=100, ge=1, le=1000)
    chunks_per_document: int = Field(default=1, ge=1)
    num_docs_per_combination: list[int] = Field(default_factory=lambda: [2, 5])
    random_seed: int = Field(default=42, ge=0)

    @field_validator("num_docs_per_combination")
    @classmethod
    def validate_doc_combination(cls, v: list[int]) -> list[int]:
        if len(v) != 2 or v[0] >= v[1] or any(x < 2 for x in v):
            raise ValueError("num_docs_per_combination must be [min, max] where min >= 2 and min < max")
        return v

    @model_validator(mode="after")
    def load_prompts(self):
        # Load prompts from files if they exist
        prompts_map = {
            "multi_hop_system_prompt": "yourbench/prompts/question_generation/multi_hop_system_prompt.md",
            "multi_hop_system_prompt_multi": "yourbench/prompts/question_generation/multi_hop_system_prompt_multi.md",
            "multi_hop_user_prompt": "yourbench/prompts/question_generation/multi_hop_user_prompt.md",
        }

        for attr, path_str in prompts_map.items():
            path = Path(path_str)
            if path.exists():
                setattr(self, attr, path.read_text(encoding="utf-8").strip())

        return self


class QuestionRewritingConfig(BaseModel):
    """Configuration for question rewriting."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    run: bool = False
    question_rewriting_system_prompt: str = Field(default="")
    question_rewriting_user_prompt: str = Field(default="")
    additional_instructions: str = (
        "Rewrite the question to sound more natural and conversational while preserving the exact meaning."
    )

    @model_validator(mode="after")
    def load_prompts(self):
        # Load prompts from files if they exist
        prompts_map = {
            "question_rewriting_system_prompt": "yourbench/prompts/question_rewriting/question_rewriting_system_prompt.md",
            "question_rewriting_user_prompt": "yourbench/prompts/question_rewriting/question_rewriting_user_prompt.md",
        }

        for attr, path_str in prompts_map.items():
            path = Path(path_str)
            if path.exists():
                setattr(self, attr, path.read_text(encoding="utf-8").strip())

        return self


class LightevalConfig(BaseModel):
    """Configuration for lighteval stages."""

    model_config = ConfigDict(validate_assignment=True)

    run: bool = False


class CitationScoreFilteringConfig(BaseModel):
    """Configuration for citation score filtering."""

    model_config = ConfigDict(validate_assignment=True)

    run: bool = False
    subset: str = "prepared_lighteval"
    alpha: float = Field(default=0.7, ge=0, le=1)
    beta: float = Field(default=0.3, ge=0, le=1)

    @model_validator(mode="after")
    def validate_alpha_beta(self):
        if abs(self.alpha + self.beta - 1.0) > 1e-6:
            raise ValueError("alpha + beta must equal 1.0")
        return self


class PipelineConfig(BaseModel):
    """Configuration for the pipeline."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",  # Allow extra fields for flexibility
    )

    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    question_generation: QuestionGenerationConfig = Field(default_factory=QuestionGenerationConfig)
    single_shot_question_generation: SingleShotQuestionGenerationConfig = Field(
        default_factory=SingleShotQuestionGenerationConfig
    )
    multi_hop_question_generation: MultiHopQuestionGenerationConfig = Field(
        default_factory=MultiHopQuestionGenerationConfig
    )
    cross_document_question_generation: CrossDocumentQuestionGenerationConfig = Field(
        default_factory=CrossDocumentQuestionGenerationConfig
    )
    question_rewriting: QuestionRewritingConfig = Field(default_factory=QuestionRewritingConfig)
    lighteval: LightevalConfig = Field(default_factory=LightevalConfig)
    prepare_lighteval: LightevalConfig = Field(default_factory=LightevalConfig)
    citation_score_filtering: CitationScoreFilteringConfig = Field(default_factory=CitationScoreFilteringConfig)


class YourbenchConfig(BaseModel):
    """The main configuration class for the YourBench pipeline."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        validate_default=True,
    )

    hf_configuration: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)
    pipeline_config: PipelineConfig = Field(default_factory=PipelineConfig)
    model_list: list[ModelConfig] = Field(default_factory=list)
    model_roles: dict[str, list[str]] = Field(default_factory=dict)
    debug: bool = False

    @model_validator(mode="after")
    def assign_default_model_roles(self):
        """Assign default model roles for each pipeline stage if not specified."""
        if not self.model_list:
            return self

        # Get the first model name as default
        default_model = self.model_list[0].model_name
        if not default_model:
            return self

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

        return self

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "YourbenchConfig":
        """
        Load YAML config with proper validation and legacy support.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e

        # Legacy compatibility: handle both 'models' and 'model_list'
        if "models" in data and "model_list" not in data:
            data["model_list"] = data.pop("models")
            logger.info("Converted legacy 'models' field to 'model_list'")

        # Handle nested pipeline configuration properly
        pipeline_data = data.get("pipeline", {})

        # Process each stage and set run=True by default if stage is present
        processed_pipeline = {}
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

        for stage_name, config_class in stage_config_classes.items():
            if stage_name in pipeline_data:
                stage_config = pipeline_data[stage_name]
                if isinstance(stage_config, dict):
                    # Set run=True by default if not specified
                    stage_config.setdefault("run", True)
                elif stage_config is None:
                    # Handle empty stage configs like "ingestion:" with no value
                    stage_config = {"run": True}

                # Create the specific config instance
                processed_pipeline[stage_name] = config_class(**stage_config)
            else:
                # Create default instance
                processed_pipeline[stage_name] = config_class()

        # Create the pipeline config
        pipeline_config = PipelineConfig(**processed_pipeline)

        # Build final config
        config_data = {
            "hf_configuration": HuggingFaceConfig(**data.get("hf_configuration", {})),
            "pipeline_config": pipeline_config,
            "model_list": [ModelConfig(**m) for m in data.get("model_list", [])],
            "model_roles": data.get("model_roles", {}),
            "debug": data.get("debug", False),
        }

        try:
            return cls(**config_data)
        except Exception as e:
            logger.error(f"Configuration validation failed for {path}: {e}")
            raise ValueError(f"Invalid configuration: {e}") from e

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, handling Path objects and other types
        config_dict = self.model_dump(mode="json", exclude_defaults=False)

        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(config_dict, fh, default_flow_style=False, indent=2, sort_keys=False)

        logger.info(f"Configuration saved to {path}")

    def model_dump_yaml(self) -> str:
        """Return configuration as YAML string."""
        config_dict = self.model_dump(mode="json", exclude_defaults=False)
        return yaml.dump(config_dict, default_flow_style=False, indent=2, sort_keys=False)


if __name__ == "__main__":
    # Test loading the simple example config
    config = YourbenchConfig.from_yaml("example/configs/simple_example.yaml")
    print(config.model_dump_yaml())
