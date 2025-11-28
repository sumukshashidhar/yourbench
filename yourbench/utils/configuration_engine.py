"""
Module handles everything related to the configuration of the pipeline.
"""

import os
from typing import TYPE_CHECKING, Any, Union, ClassVar
from pathlib import Path
from importlib.resources import files

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
from yourbench.utils.url_utils import get_api_key_for_url, validate_api_key_for_url


if TYPE_CHECKING:
    pass


def _expand_env(value: str) -> str:
    """
    Replace leading '$VARNAME' with its environment value.
    Special case: if $HF_ORGANIZATION is missing we try HF_TOKEN + whoami().
    """
    if not (isinstance(value, str) and value.startswith("$")):
        return value

    var = value[1:]
    env = os.getenv(var)
    if env is not None:
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


def _load_prompt_from_package(package_path: str) -> str | None:
    """
    Load prompt content from package resources using importlib.resources.

    Args:
        package_path: Path relative to yourbench.prompts (e.g., "ingestion/pdf_llm_prompt.md")

    Returns:
        Prompt content if found, None otherwise
    """
    try:
        # Access the prompts package
        prompts_files = files("yourbench.prompts")

        # Navigate to the specific file
        parts = package_path.split("/")
        current_files = prompts_files

        for part in parts[:-1]:  # Navigate to subdirectories
            current_files = current_files / part

        # Get the file content
        file_resource = current_files / parts[-1]
        if file_resource.is_file():
            content = file_resource.read_text(encoding="utf-8").strip()
            logger.debug(f"Loaded prompt from package: {package_path}")
            return content
        else:
            logger.debug(f"Prompt file not found in package: {package_path}")
            return None

    except Exception as e:
        logger.debug(f"Failed to load prompt from package {package_path}: {e}")
        return None


def _load_prompt_or_string(value: str, default_fallback: str = "") -> str:
    """
    Load prompt content from file path, use as string, or fall back to default.

    This function now prioritizes loading from package resources for prompts
    in the yourbench.prompts package, which ensures compatibility when the
    package is installed via pip.

    Args:
        value: Prompt value - can be file path, string content, or empty
        default_fallback: Default prompt to use if value is empty

    Returns:
        Final prompt content
    """
    if not value:
        return default_fallback

    # If it's multi-line or very long, it's almost certainly string content, not a path
    if "\n" in value or len(value) > 300:
        return value

    # Check if it looks like a file path
    value_path = Path(value)

    # If it has common text file extensions, try to load it as a file
    text_extensions = {".md", ".txt", ".prompt", ".text"}
    if value_path.suffix.lower() in text_extensions:
        # First, try to load from package resources if it looks like a yourbench prompt
        if str(value_path).startswith("yourbench/prompts/"):
            package_path = str(value_path)[len("yourbench/prompts/") :]
            package_content = _load_prompt_from_package(package_path)
            if package_content is not None:
                return package_content

        # Fallback to file system loading (for development and custom prompts)
        try:
            if value_path.exists():
                content = value_path.read_text(encoding="utf-8").strip()
                logger.debug(f"Loaded prompt from file: {value_path}")
                return content
            else:
                logger.warning(f"Prompt file not found: {value_path}, treating as string content")
                return value
        except Exception as e:
            logger.warning(f"Failed to read prompt file {value_path}: {e}, treating as string content")
            return value

    # Otherwise, treat as string content
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
    export_jsonl: bool = False
    jsonl_export_dir: Path | None = Path("data/jsonl_export")
    push_to_hub: bool = True

    @field_validator("hf_organization", "hf_token")
    @classmethod
    def expand_env_vars(cls, v: str) -> str:
        return _expand_env(v)

    @model_validator(mode="after")
    def expand_all_env_vars(self):
        # Use object.__setattr__ to bypass validation and avoid recursion
        object.__setattr__(self, "hf_token", _expand_env(self.hf_token))
        object.__setattr__(self, "hf_organization", _expand_env(self.hf_organization))
        return self

    @model_validator(mode="after")
    def validate_required_tokens(self):
        """Validate that required tokens are set when needed."""
        # Check if HF_TOKEN is needed and set
        if self.hf_token == "$HF_TOKEN" and not os.getenv("HF_TOKEN"):
            logger.warning("HF_TOKEN environment variable not set. Please set it with: export HF_TOKEN=your_token")
        return self

    @field_validator("local_dataset_dir", "jsonl_export_dir")
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
    extra_parameters: dict[str, Any] = Field(default_factory=dict)

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
        # Also ensure api_key is expanded
        if self.api_key:
            object.__setattr__(self, "api_key", _expand_env(self.api_key))
        return self

    @model_validator(mode="after")
    def validate_api_keys(self):
        """Validate that required API keys are set based on base_url."""
        # Use the shared validation function
        is_valid, error_msg = validate_api_key_for_url(self.base_url, self.api_key, self.model_name)
        if not is_valid:
            logger.error(error_msg)
            raise ValueError(error_msg)
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
        # Load PDF LLM prompt using package resources first, then fallback
        default_fallback = ""

        # Try to load from package resources first
        package_content = _load_prompt_from_package("ingestion/pdf_llm_prompt.md")
        if package_content is not None:
            default_fallback = package_content
        else:
            # Fallback to file system (for development)
            default_prompt_path = "yourbench/prompts/ingestion/pdf_llm_prompt.md"
            if Path(default_prompt_path).exists():
                try:
                    default_fallback = Path(default_prompt_path).read_text(encoding="utf-8").strip()
                except Exception:
                    pass

        # Use object.__setattr__ to bypass validation and avoid recursion
        object.__setattr__(self, "pdf_llm_prompt", _load_prompt_or_string(self.pdf_llm_prompt, default_fallback))

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
        # Load default prompts from markdown files
        summarization_default = ""
        combine_summaries_default = ""

        # Try to load from package resources first
        summarization_content = _load_prompt_from_package("summarization/summarization_user_prompt.md")
        if summarization_content is not None:
            summarization_default = summarization_content

        combine_summaries_content = _load_prompt_from_package("summarization/combine_summaries_user_prompt.md")
        if combine_summaries_content is not None:
            combine_summaries_default = combine_summaries_content

        # Load prompts: can be file paths, string content, or fall back to defaults
        # Use object.__setattr__ to bypass validation and avoid recursion
        object.__setattr__(
            self,
            "summarization_user_prompt",
            _load_prompt_or_string(self.summarization_user_prompt, summarization_default),
        )
        object.__setattr__(
            self,
            "combine_summaries_user_prompt",
            _load_prompt_or_string(self.combine_summaries_user_prompt, combine_summaries_default),
        )

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
        # Load default prompts from markdown files
        system_prompt_default = ""
        system_prompt_multi_default = ""
        user_prompt_default = ""

        # Load prompts from package resources
        system_content = _load_prompt_from_package("question_generation/single_shot_system_prompt.md")
        if system_content is not None:
            system_prompt_default = system_content

        system_multi_content = _load_prompt_from_package("question_generation/single_shot_system_prompt_multi.md")
        if system_multi_content is not None:
            system_prompt_multi_default = system_multi_content

        user_content = _load_prompt_from_package("question_generation/single_shot_user_prompt.md")
        if user_content is not None:
            user_prompt_default = user_content

        # Load prompts: can be file paths, string content, or fall back to defaults
        # Use object.__setattr__ to bypass validation and avoid recursion
        object.__setattr__(
            self,
            "single_shot_system_prompt",
            _load_prompt_or_string(self.single_shot_system_prompt, system_prompt_default),
        )
        object.__setattr__(
            self,
            "single_shot_system_prompt_multi",
            _load_prompt_or_string(self.single_shot_system_prompt_multi, system_prompt_multi_default),
        )
        object.__setattr__(
            self,
            "single_shot_user_prompt",
            _load_prompt_or_string(self.single_shot_user_prompt, user_prompt_default),
        )

        return self


class MultiHopQuestionGenerationConfig(QuestionGenerationConfig):
    """Configuration for multi-hop question generation."""

    multi_hop_system_prompt: str = Field(default="")
    multi_hop_system_prompt_multi: str = Field(default="")
    multi_hop_user_prompt: str = Field(default="")

    @model_validator(mode="after")
    def load_prompts(self):
        # Load default prompts from markdown files
        system_prompt_default = ""
        system_prompt_multi_default = ""
        user_prompt_default = ""

        # Load prompts from package resources
        system_content = _load_prompt_from_package("question_generation/multi_hop_system_prompt.md")
        if system_content is not None:
            system_prompt_default = system_content

        # For multi-choice, we need to check if there's a separate multi-hop multi-choice prompt
        # If not, we'll use the same as the regular multi-hop system prompt
        system_prompt_multi_default = system_prompt_default  # Default to same as regular

        user_content = _load_prompt_from_package("question_generation/multi_hop_user_prompt.md")
        if user_content is not None:
            user_prompt_default = user_content

        # Load prompts: can be file paths, string content, or fall back to defaults
        # Use object.__setattr__ to bypass validation and avoid recursion
        object.__setattr__(
            self,
            "multi_hop_system_prompt",
            _load_prompt_or_string(self.multi_hop_system_prompt, system_prompt_default),
        )
        object.__setattr__(
            self,
            "multi_hop_system_prompt_multi",
            _load_prompt_or_string(self.multi_hop_system_prompt_multi, system_prompt_multi_default),
        )
        object.__setattr__(
            self,
            "multi_hop_user_prompt",
            _load_prompt_or_string(self.multi_hop_user_prompt, user_prompt_default),
        )

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
        # Load default prompts from markdown files
        system_prompt_default = ""
        system_prompt_multi_default = ""
        user_prompt_default = ""

        # Load prompts from package resources
        system_content = _load_prompt_from_package("question_generation/multi_hop_system_prompt.md")
        if system_content is not None:
            system_prompt_default = system_content

        # For multi-choice, we need to check if there's a separate multi-hop multi-choice prompt
        # If not, we'll use the same as the regular multi-hop system prompt
        system_prompt_multi_default = system_prompt_default  # Default to same as regular

        user_content = _load_prompt_from_package("question_generation/multi_hop_user_prompt.md")
        if user_content is not None:
            user_prompt_default = user_content

        # Load prompts: can be file paths, string content, or fall back to defaults
        # Use object.__setattr__ to bypass validation and avoid recursion
        object.__setattr__(
            self,
            "multi_hop_system_prompt",
            _load_prompt_or_string(self.multi_hop_system_prompt, system_prompt_default),
        )
        object.__setattr__(
            self,
            "multi_hop_system_prompt_multi",
            _load_prompt_or_string(self.multi_hop_system_prompt_multi, system_prompt_multi_default),
        )
        object.__setattr__(
            self,
            "multi_hop_user_prompt",
            _load_prompt_or_string(self.multi_hop_user_prompt, user_prompt_default),
        )

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
        # Load default prompts from markdown files
        system_prompt_default = ""
        user_prompt_default = ""

        # Load prompts from package resources
        system_content = _load_prompt_from_package("question_rewriting/question_rewriting_system_prompt.md")
        if system_content is not None:
            system_prompt_default = system_content

        user_content = _load_prompt_from_package("question_rewriting/question_rewriting_user_prompt.md")
        if user_content is not None:
            user_prompt_default = user_content

        # Load prompts: can be file paths, string content, or fall back to defaults
        # Use object.__setattr__ to bypass validation and avoid recursion
        object.__setattr__(
            self,
            "question_rewriting_system_prompt",
            _load_prompt_or_string(self.question_rewriting_system_prompt, system_prompt_default),
        )
        object.__setattr__(
            self,
            "question_rewriting_user_prompt",
            _load_prompt_or_string(self.question_rewriting_user_prompt, user_prompt_default),
        )

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

    # Define pipeline stages in execution order
    STAGE_ORDER: ClassVar[list[str]] = [
        "ingestion",
        "summarization",
        "chunking",
        "question_generation",
        "single_shot_question_generation",
        "multi_hop_question_generation",
        "cross_document_question_generation",
        "question_rewriting",
        "prepare_lighteval",
        "lighteval",
        "citation_score_filtering",
    ]

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

    def get_enabled_stages(self) -> list[str]:
        """Return list of enabled stages in execution order."""
        return [stage for stage in self.STAGE_ORDER if getattr(self, stage).run]

    def get_stage_config(self, stage_name: str):
        """Get configuration for a specific stage."""
        if stage_name not in self.STAGE_ORDER:
            raise ValueError(f"Unknown stage: {stage_name}")
        return getattr(self, stage_name)


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

        # Use the pipeline's ordered stage list for consistency
        pipeline_stages = self.pipeline_config.STAGE_ORDER

        # Assign default model to stages that don't have model roles defined
        for stage in pipeline_stages:
            if stage not in self.model_roles:
                self.model_roles[stage] = [default_model]

        return self

    def get_model_for_stage(self, stage_name: str) -> str | None:
        """Get the model assigned to a specific stage."""
        if stage_name not in self.model_roles:
            return None
        return self.model_roles[stage_name][0] if self.model_roles[stage_name] else None

    def is_stage_enabled(self, stage_name: str) -> bool:
        """Check if a pipeline stage is enabled."""
        try:
            stage_config = self.pipeline_config.get_stage_config(stage_name)
            return stage_config.run
        except (AttributeError, ValueError):
            return False

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file with proper key names."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, handling Path objects and other types
        config_dict = self.model_dump(mode="json", exclude_defaults=False)

        # Rename pipeline_config to pipeline for compatibility with from_yaml
        if "pipeline_config" in config_dict:
            config_dict["pipeline"] = config_dict.pop("pipeline_config")

        # Preserve environment variable references in HF configuration
        if "hf_configuration" in config_dict:
            hf_conf = config_dict["hf_configuration"]
            # Always use environment variable references for sensitive data
            if "hf_token" in hf_conf:
                hf_conf["hf_token"] = "$HF_TOKEN"
            if "hf_organization" in hf_conf and hf_conf.get("hf_organization"):
                # Only preserve if it looks like it was originally an env var
                if not hf_conf["hf_organization"].startswith("$"):
                    # If we got an actual org name from whoami, keep it as $HF_ORGANIZATION
                    hf_conf["hf_organization"] = "$HF_ORGANIZATION"

        # Preserve environment variable references in model configurations
        if "model_list" in config_dict:
            for model in config_dict["model_list"]:
                if "api_key" in model:
                    # Determine the appropriate API key env var based on base_url
                    base_url = model.get("base_url", "")
                    model["api_key"] = get_api_key_for_url(base_url)

        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(config_dict, fh, default_flow_style=False, indent=2, sort_keys=False)
        logger.info(f"Configuration saved to {path}")

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
        if pipeline_data is None:
            pipeline_data = {}

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
        # Filter out None values from hf_configuration
        hf_config_data = data.get("hf_configuration", {})
        if hf_config_data:
            hf_config_data = {k: v for k, v in hf_config_data.items() if v is not None}

        config_data = {
            "hf_configuration": HuggingFaceConfig(**hf_config_data),
            "pipeline_config": pipeline_config,
            "model_list": [
                ModelConfig(**{k: v for k, v in m.items() if v is not None}) for m in data.get("model_list", [])
            ],
            "model_roles": data.get("model_roles", {}),
            "debug": data.get("debug", False),
        }

        try:
            return cls(**config_data)
        except Exception as e:
            logger.error(f"Configuration validation failed for {path}: {e}")
            raise ValueError(f"Invalid configuration: {e}") from e

    def model_dump_yaml(self) -> str:
        """Return configuration as YAML string."""
        config_dict = self.model_dump(mode="json", exclude_defaults=False)
        return yaml.dump(config_dict, default_flow_style=False, indent=2, sort_keys=False)


def is_yourbench_config(config: any) -> bool:
    """Type-safe check if config is a YourbenchConfig instance."""
    return isinstance(config, YourbenchConfig)


if __name__ == "__main__":
    # Test loading the simple example config
    config = YourbenchConfig.from_yaml("example/configs/simple_example.yaml")
    logger.info(config.model_dump_yaml())
