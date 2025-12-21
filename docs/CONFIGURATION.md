# YourBench Configuration Guide

YourBench uses YAML configuration files to define your pipeline settings. This guide covers how to configure each component.

## Table of Contents

- [Overview](#overview)
- [Configuration Structure](#configuration-structure)
- [Core Components](#core-components)
  - [Hugging Face Configuration](#hugging-face-configuration)
  - [Model Configuration](#model-configuration)
  - [Pipeline Configuration](#pipeline-configuration)
- [Pipeline Stages](#pipeline-stages)
  - [Ingestion](#ingestion)
  - [Summarization](#summarization)
  - [Chunking](#chunking)
  - [Question Generation](#question-generation)
  - [Question Rewriting](#question-rewriting)
  - [LightEval](#lighteval)
  - [Citation Score Filtering](#citation-score-filtering)
- [Advanced Features](#advanced-features)
  - [Environment Variables](#environment-variables)
  - [Custom Prompts](#custom-prompts)
  - [Custom Question Schemas](#custom-question-schemas)
  - [Model Role Assignment](#model-role-assignment)
- [Configuration Examples](#configuration-examples)

## Overview

YourBench configuration is structured around three main components:

1. **Hugging Face Configuration** - Settings for dataset management and uploads
2. **Model Configuration** - LLM provider settings and API credentials
3. **Pipeline Configuration** - Stage-specific settings for the processing pipeline

## Configuration Structure

A basic config file looks like this:

```yaml
hf_configuration:
  hf_dataset_name: my-dataset-name

model_list:
  - model_name: zai-org/GLM-4.5

pipeline:
  ingestion:
    source_documents_dir: data/raw
  summarization:
  chunking:
  single_shot_question_generation:
  prepare_lighteval:
```

Key points:
- **Pipeline stages are enabled by presence** - if a stage appears in `pipeline:`, it runs
- **Environment variables** can be used with `$VAR_NAME` syntax
- **Model roles** are auto-assigned if not specified (uses first model in `model_list`)

## Core Components

### Hugging Face Configuration

Controls dataset naming, organization, and upload behavior.

```yaml
hf_configuration:
  hf_dataset_name: my-dataset-name  # Required: dataset name on Hub
  hf_organization: $HF_ORGANIZATION  # Optional: organization name
  hf_token: $HF_TOKEN               # Optional: HF API token (or set env var)
  private: false                    # Default: false - dataset visibility
  concat_if_exist: false           # Default: false - append to existing dataset
  local_dataset_dir: data/saved_dataset  # Default: local save path
  local_saving: true               # Default: true - save locally
  upload_card: true                # Default: true - upload dataset card
  export_jsonl: false              # Default: false - export as JSONL
  jsonl_export_dir: data/jsonl     # JSONL export directory
```

### Model Configuration

Define LLMs used by the pipeline:

```yaml
model_list:
  - model_name: zai-org/GLM-4.5    # Required: model name or HF model ID
    base_url: null                  # Optional: custom API endpoint
    api_key: $HF_TOKEN             # Optional: API key (defaults to HF_TOKEN)
    max_concurrent_requests: 32    # Default: 32 - parallel request limit
    encoding_name: cl100k_base     # Default: tokenizer for counting
    provider: null                 # Optional: openai, anthropic, etc.
    bill_to: null                  # Optional: billing project
    extra_parameters: {}           # Optional: provider-specific params
```

Multiple models can be defined and assigned to different pipeline stages.

### Pipeline Configuration

Each stage can be enabled by including it in the `pipeline:` section:

```yaml
pipeline:
  ingestion:
    source_documents_dir: data/raw    # Required: input documents
    output_dir: data/processed        # Default: processed output
  summarization:
    max_tokens: 32768                 # Max tokens per summary chunk
  chunking:
    l_max_tokens: 8192               # Max tokens per chunk
  single_shot_question_generation:
    question_mode: open-ended         # or multi-choice
  prepare_lighteval:                  # Just include to enable
```

## Pipeline Stages

### Ingestion

Converts source documents to markdown format.

```yaml
pipeline:
  ingestion:
    source_documents_dir: data/raw           # Required
    output_dir: data/processed               # Default: data/processed
    upload_to_hub: true                      # Default: true
    llm_ingestion: false                     # Use LLM for PDF processing
    pdf_dpi: 300                             # DPI for PDF rendering
    pdf_llm_prompt: path/to/prompt.md       # Custom PDF extraction prompt
    supported_file_extensions: [".md", ".txt", ".pdf"]  # Default
```

### Summarization

Creates summaries of processed documents.

```yaml
pipeline:
  summarization:
    max_tokens: 32768           # Max tokens per chunk
    token_overlap: 512          # Overlap between chunks
    encoding_name: cl100k_base  # Tokenizer
    summarization_user_prompt: path/to/prompt.md
    combine_summaries_user_prompt: path/to/combine_prompt.md
```

### Chunking

Splits documents into chunks for question generation.

```yaml
pipeline:
  chunking:
    l_max_tokens: 8192     # Max tokens per chunk
    token_overlap: 512     # Overlap between chunks
    encoding_name: cl100k_base
    h_min: 2               # Min chunks for multi-hop
    h_max: 5               # Max chunks for multi-hop
    num_multihops_factor: 1
```

### Question Generation

Generate questions from document chunks. Three types are available:

#### Single-Shot Questions

```yaml
pipeline:
  single_shot_question_generation:
    question_mode: open-ended       # or multi-choice
    additional_instructions: ""     # Extra context for LLM
    question_schema: path/to/schema.py  # Optional: custom output format
    single_shot_system_prompt: path/to/prompt.md
    single_shot_user_prompt: path/to/prompt.md
    chunk_sampling:
      enable: false
      num_samples: 100
      strategy: random              # or uniform
      random_seed: 42
```

#### Multi-Hop Questions

```yaml
pipeline:
  multi_hop_question_generation:
    question_mode: open-ended
    additional_instructions: ""
    question_schema: path/to/schema.py  # Optional: custom output format
    multi_hop_system_prompt: path/to/prompt.md
    multi_hop_user_prompt: path/to/prompt.md
```

#### Cross-Document Questions

```yaml
pipeline:
  cross_document_question_generation:
    question_mode: open-ended
    additional_instructions: ""
    question_schema: path/to/schema.py  # Optional: custom output format
    max_combinations: 100
    chunks_per_document: 1
    num_docs_per_combination: [2, 5]
    random_seed: 42
```

### Question Rewriting

Rewrites generated questions for clarity/style.

```yaml
pipeline:
  question_rewriting:
    question_rewriting_system_prompt: path/to/prompt.md
    question_rewriting_user_prompt: path/to/prompt.md
    additional_instructions: ""
```

### LightEval

Prepares dataset for evaluation.

```yaml
pipeline:
  prepare_lighteval:    # Just include to enable
```

### Citation Score Filtering

Computes citation scores for filtering.

```yaml
pipeline:
  citation_score_filtering:
    subset: prepared_lighteval  # Dataset subset to process
    alpha: 0.7                  # Weight for chunk overlap
    beta: 0.3                   # Weight for answer overlap
```

## Advanced Features

### Environment Variables

Use `$VAR_NAME` syntax to reference environment variables:

```yaml
hf_configuration:
  hf_token: $HF_TOKEN
  hf_organization: $HF_ORGANIZATION

model_list:
  - model_name: gpt-4
    api_key: $OPENAI_API_KEY
    base_url: https://api.openai.com/v1
```

### Custom Prompts

Point to custom prompt files:

```yaml
pipeline:
  ingestion:
    pdf_llm_prompt: prompts/custom_pdf_prompt.md
  summarization:
    summarization_user_prompt: prompts/summary.md
```

Prompts are loaded from:
1. The specified file path (if exists)
2. Package defaults (built-in prompts)

### Custom Question Schemas

Define custom output formats for generated questions using Pydantic models.

**Config:**
```yaml
pipeline:
  single_shot_question_generation:
    question_schema: ./schemas/my_schema.py
```

**Schema file (must export `DataFormat` class):**
```python
# my_schema.py
from pydantic import BaseModel, Field
from typing import Literal

class DataFormat(BaseModel):
    question: str = Field(description="The question text")
    answer: str = Field(description="Complete answer")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Difficulty level")
    citations: list[str] = Field(description="Source quotes")
```

**Key rules:**
- Schema file must contain a class named `DataFormat`
- Class must inherit from Pydantic `BaseModel`
- Use `Field(description=...)` to guide the LLM
- Custom fields are automatically preserved in output

**Field aliasing:** Certain fields are automatically mapped:
- `reasoning`, `explanation` → `thought_process`
- String `difficulty` (easy/medium/hard) → integer `estimated_difficulty` (1-10)

See [Custom Schemas Guide](./CUSTOM_SCHEMAS.md) for detailed examples.

### Model Role Assignment

Assign specific models to pipeline stages:

```yaml
model_list:
  - model_name: gpt-4
    base_url: https://api.openai.com/v1
    api_key: $OPENAI_API_KEY
  - model_name: claude-3-opus
    base_url: https://api.anthropic.com/v1
    api_key: $ANTHROPIC_API_KEY

model_roles:
  ingestion: [gpt-4]
  summarization: [gpt-4]
  single_shot_question_generation: [claude-3-opus]
  multi_hop_question_generation: [claude-3-opus]
```

If `model_roles` is not specified, all stages use the first model in `model_list`.

## Configuration Examples

### Minimal Config

```yaml
hf_configuration:
  hf_dataset_name: my-benchmark

model_list:
  - model_name: zai-org/GLM-4.5

pipeline:
  ingestion:
    source_documents_dir: data/raw
  summarization:
  chunking:
  single_shot_question_generation:
  prepare_lighteval:
```

### Full Pipeline with Multi-Hop

```yaml
hf_configuration:
  hf_dataset_name: comprehensive-benchmark
  hf_organization: $HF_ORGANIZATION
  private: true
  local_saving: true
  export_jsonl: true
  jsonl_export_dir: output/jsonl

model_list:
  - model_name: gpt-4-turbo
    base_url: https://api.openai.com/v1
    api_key: $OPENAI_API_KEY
    max_concurrent_requests: 16

pipeline:
  ingestion:
    source_documents_dir: data/raw
    output_dir: data/processed
    llm_ingestion: true
    pdf_dpi: 300

  summarization:
    max_tokens: 32768
    token_overlap: 512

  chunking:
    l_max_tokens: 8192
    h_min: 2
    h_max: 5

  single_shot_question_generation:
    question_mode: open-ended
    additional_instructions: "Focus on technical details"

  multi_hop_question_generation:
    question_mode: open-ended

  question_rewriting:

  prepare_lighteval:

  citation_score_filtering:
    alpha: 0.7
    beta: 0.3

debug: false
```

### Custom Schema Example

```yaml
hf_configuration:
  hf_dataset_name: technical-benchmark

model_list:
  - model_name: gpt-4-turbo
    base_url: $OPENAI_BASE_URL
    api_key: $OPENAI_API_KEY

pipeline:
  ingestion:
    source_documents_dir: data/raw
  chunking:
  single_shot_question_generation:
    question_schema: ./schemas/technical.py
    additional_instructions: "Focus on implementation details"
  prepare_lighteval:
```

### OpenAI-Compatible Provider

```yaml
hf_configuration:
  hf_dataset_name: custom-provider-test

model_list:
  - model_name: my-local-model
    base_url: http://localhost:8000/v1
    api_key: not-needed
    max_concurrent_requests: 8

pipeline:
  ingestion:
    source_documents_dir: data/raw
  summarization:
  chunking:
  single_shot_question_generation:
  prepare_lighteval:
```
