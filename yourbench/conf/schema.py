"""
Structured config schemas with defaults for YourBench.

These dataclasses define the expected config structure and provide
sensible defaults. OmegaConf merges user configs with these defaults.
"""

from dataclasses import field, dataclass


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""

    pass


@dataclass
class HFConfig:
    """HuggingFace dataset configuration."""

    hf_dataset_name: str = ""
    hf_organization: str = ""
    hf_token: str = ""
    private: bool = False
    concat_if_exist: bool = False
    local_dataset_dir: str = "data/saved_dataset"
    local_saving: bool = True
    upload_card: bool = True
    export_jsonl: bool = False
    jsonl_export_dir: str = "data/jsonl_export"
    push_to_hub: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str = ""
    base_url: str | None = None
    api_key: str | None = None
    max_concurrent_requests: int = 32
    encoding_name: str = "cl100k_base"
    provider: str | None = None
    bill_to: str | None = None
    extra_parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.max_concurrent_requests < 1:
            raise ConfigValidationError(f"max_concurrent_requests must be >= 1, got {self.max_concurrent_requests}")


@dataclass
class ChunkSamplingConfig:
    """Chunk sampling configuration."""

    enable: bool = False
    num_samples: int = 100
    strategy: str = "random"
    random_seed: int = 42

    def __post_init__(self):
        if self.num_samples < 1:
            raise ConfigValidationError(f"num_samples must be >= 1, got {self.num_samples}")


@dataclass
class IngestionConfig:
    """Ingestion stage configuration."""

    run: bool = False
    source_documents_dir: str = "data/raw"
    output_dir: str = "data/processed"
    upload_to_hub: bool = True
    llm_ingestion: bool = False
    pdf_dpi: int = 300
    pdf_llm_prompt: str = ""
    supported_file_extensions: list = field(default_factory=lambda: [".md", ".txt", ".pdf"])


@dataclass
class SummarizationConfig:
    """Summarization stage configuration."""

    run: bool = False
    max_tokens: int = 32768
    token_overlap: int = 512
    encoding_name: str = "cl100k_base"
    summarization_user_prompt: str = ""
    combine_summaries_user_prompt: str = ""

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ConfigValidationError(f"max_tokens must be > 0, got {self.max_tokens}")
        if self.token_overlap < 0:
            raise ConfigValidationError(f"token_overlap must be >= 0, got {self.token_overlap}")
        if self.token_overlap >= self.max_tokens:
            raise ConfigValidationError(
                f"token_overlap ({self.token_overlap}) must be < max_tokens ({self.max_tokens})"
            )


@dataclass
class ChunkingConfig:
    """Chunking stage configuration."""

    run: bool = False
    l_max_tokens: int = 8192
    token_overlap: int = 512
    encoding_name: str = "cl100k_base"
    h_min: int = 2
    h_max: int = 5
    num_multihops_factor: int = 1

    def __post_init__(self):
        if self.l_max_tokens <= 0:
            raise ConfigValidationError(f"l_max_tokens must be > 0, got {self.l_max_tokens}")
        if self.token_overlap < 0:
            raise ConfigValidationError(f"token_overlap must be >= 0, got {self.token_overlap}")
        if self.h_min < 1:
            raise ConfigValidationError(f"h_min must be >= 1, got {self.h_min}")
        if self.h_max < self.h_min:
            raise ConfigValidationError(f"h_max ({self.h_max}) must be >= h_min ({self.h_min})")
        if self.num_multihops_factor < 1:
            raise ConfigValidationError(f"num_multihops_factor must be >= 1, got {self.num_multihops_factor}")


@dataclass
class SingleShotConfig:
    """Single-shot question generation configuration."""

    run: bool = False
    question_mode: str = "open-ended"
    additional_instructions: str = ""
    single_shot_system_prompt: str = ""
    single_shot_system_prompt_multi: str = ""
    single_shot_user_prompt: str = ""
    chunk_sampling: ChunkSamplingConfig = field(default_factory=ChunkSamplingConfig)

    def __post_init__(self):
        valid_modes = {"open-ended", "multi-choice", ""}
        mode = self.question_mode.strip().lower() if self.question_mode else ""
        if mode and mode not in valid_modes:
            raise ConfigValidationError(
                f"question_mode must be 'open-ended' or 'multi-choice', got '{self.question_mode}'"
            )


@dataclass
class MultiHopConfig:
    """Multi-hop question generation configuration."""

    run: bool = False
    question_mode: str = "open-ended"
    additional_instructions: str = ""
    multi_hop_system_prompt: str = ""
    multi_hop_system_prompt_multi: str = ""
    multi_hop_user_prompt: str = ""

    def __post_init__(self):
        valid_modes = {"open-ended", "multi-choice", ""}
        mode = self.question_mode.strip().lower() if self.question_mode else ""
        if mode and mode not in valid_modes:
            raise ConfigValidationError(
                f"question_mode must be 'open-ended' or 'multi-choice', got '{self.question_mode}'"
            )


@dataclass
class CrossDocConfig:
    """Cross-document question generation configuration."""

    run: bool = False
    question_mode: str = "open-ended"
    additional_instructions: str = ""
    multi_hop_system_prompt: str = ""
    multi_hop_system_prompt_multi: str = ""
    multi_hop_user_prompt: str = ""
    max_combinations: int = 100
    chunks_per_document: int = 1
    num_docs_per_combination: list = field(default_factory=lambda: [2, 5])
    random_seed: int = 42

    def __post_init__(self):
        valid_modes = {"open-ended", "multi-choice", ""}
        mode = self.question_mode.strip().lower() if self.question_mode else ""
        if mode and mode not in valid_modes:
            raise ConfigValidationError(
                f"question_mode must be 'open-ended' or 'multi-choice', got '{self.question_mode}'"
            )
        if self.max_combinations < 1:
            raise ConfigValidationError(f"max_combinations must be >= 1, got {self.max_combinations}")
        if self.chunks_per_document < 1:
            raise ConfigValidationError(f"chunks_per_document must be >= 1, got {self.chunks_per_document}")
        if not isinstance(self.num_docs_per_combination, list) or len(self.num_docs_per_combination) != 2:
            raise ConfigValidationError(
                f"num_docs_per_combination must be a list of 2 elements, got {self.num_docs_per_combination}"
            )
        min_docs, max_docs = self.num_docs_per_combination
        if min_docs < 2:
            raise ConfigValidationError(f"num_docs_per_combination[0] must be >= 2, got {min_docs}")
        if max_docs < min_docs:
            raise ConfigValidationError(f"num_docs_per_combination[1] ({max_docs}) must be >= [0] ({min_docs})")


@dataclass
class QuestionRewritingConfig:
    """Question rewriting configuration."""

    run: bool = False
    question_rewriting_system_prompt: str = ""
    question_rewriting_user_prompt: str = ""
    additional_instructions: str = ""


@dataclass
class LightevalConfig:
    """Lighteval preparation configuration."""

    run: bool = False
    single_shot_subset: str = "single_shot_questions"
    multi_hop_subset: str = "multi_hop_questions"
    cross_doc_subset: str = "cross_document_questions"
    chunked_subset: str = "chunked"
    summarized_subset: str = "summarized"
    output_subset: str = "prepared_lighteval"


@dataclass
class CitationFilteringConfig:
    """Citation score filtering configuration."""

    run: bool = False
    subset: str = "prepared_lighteval"
    alpha: float = 0.7
    beta: float = 0.3

    def __post_init__(self):
        if not (0.0 <= self.alpha <= 1.0):
            raise ConfigValidationError(f"alpha must be in [0, 1], got {self.alpha}")
        if not (0.0 <= self.beta <= 1.0):
            raise ConfigValidationError(f"beta must be in [0, 1], got {self.beta}")


@dataclass
class PipelineConfig:
    """Pipeline configuration with all stages."""

    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    single_shot_question_generation: SingleShotConfig = field(default_factory=SingleShotConfig)
    multi_hop_question_generation: MultiHopConfig = field(default_factory=MultiHopConfig)
    cross_document_question_generation: CrossDocConfig = field(default_factory=CrossDocConfig)
    question_rewriting: QuestionRewritingConfig = field(default_factory=QuestionRewritingConfig)
    prepare_lighteval: LightevalConfig = field(default_factory=LightevalConfig)
    citation_score_filtering: CitationFilteringConfig = field(default_factory=CitationFilteringConfig)


@dataclass
class YourbenchConfig:
    """Root configuration schema."""

    hf_configuration: HFConfig = field(default_factory=HFConfig)
    model_list: list = field(default_factory=list)
    model_roles: dict = field(default_factory=dict)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    debug: bool = False
