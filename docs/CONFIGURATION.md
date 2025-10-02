# YourBench Configuration Guide

YourBench uses a flexible YAML-based configuration system built on Pydantic models. This guide covers all configuration options and usage patterns.

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
  - [Model Role Assignment](#model-role-assignment)
- [Configuration Examples](#configuration-examples)
- [API Reference](#api-reference)

## Overview

YourBench configuration is structured around three main components:

1. **Hugging Face Configuration** - Settings for dataset management and uploads
2. **Model Configuration** - LLM provider settings and API credentials
3. **Pipeline Configuration** - Stage-specific settings for the processing pipeline

## Configuration Structure

A typical YourBench configuration file follows this structure:

```yaml
hf_configuration:
  # Hugging Face dataset settings

model_list:
  # List of model configurations

model_roles:
  # Optional: Assign specific models to pipeline stages

pipeline:
  # Pipeline stage configurations
```

## Core Components

### Hugging Face Configuration

Controls dataset naming, organization, and upload behavior.

```yaml
hf_configuration:
  hf_dataset_name: my-dataset-name  # Default: random name
  hf_organization: $HF_ORGANIZATION  # Environment variable
  hf_token: $HF_TOKEN               # Environment variable
  private: false                    # Dataset visibility
  concat_if_exist: false           # Append to existing dataset
  local_dataset_dir: data/saved_dataset  # Local save path
  local_saving: true               # Save locally
  upload_card: true                # Upload dataset card
```

**Key Features:**
- Environment variable expansion with `$VARNAME` syntax
- Automatic HF organization detection from token if `$HF_ORGANIZATION` is not set
- Random dataset names generated if not specified

### Model Configuration

Defines LLM providers and connection settings.

```yaml
model_list:
  - model_name: gpt-4o
    base_url: https://api.openai.com/v1/
    api_key: $OPENAI_API_KEY
    max_concurrent_requests: 32
    encoding_name: cl100k_base
    provider: auto  # Automatically set if not specified
    bill_to: optional-billing-id
    extra_parameters:
      reasoning:
        effort: medium
```

**Supported Fields:**
- `model_name`: Model identifier (required)
- `base_url`: API endpoint (optional)
- `api_key`: Authentication token (supports env vars)
- `max_concurrent_requests`: Rate limiting (1-100)
- `encoding_name`: Tokenizer encoding
- `provider`: Provider type (auto-detected if not set)
- `extra_parameters`: Arbitrary JSON payload merged into every chat completion request (useful for provider-specific features like OpenRouter `reasoning` objects)

You can also provide these parameters from the CLI when running against a folder of documents using:

```bash
yourbench run ./docs --model openrouter/your-model \
  --base-url https://openrouter.ai/api/v1 \
  --model-extra-parameters '{"reasoning": {"effort": "medium"}}'
```

The flag accepts either an inline JSON object or a path to a JSON file.

### Pipeline Configuration

Controls which stages run and their parameters.

```yaml
pipeline:
  ingestion:
    run: true  # Automatically set when stage is present
    # ... stage-specific config
  
  summarization:
    # Presence implies run: true
    # ... stage-specific config
```

## Pipeline Stages

### Ingestion

Processes raw documents into structured markdown format.

```yaml
pipeline:
  ingestion:
    source_documents_dir: path/to/documents
    output_dir: path/to/output
    upload_to_hub: true
    llm_ingestion: false  # Use LLM for PDF extraction
    pdf_dpi: 300         # PDF rendering quality (72-600)
    pdf_llm_prompt: ""   # Custom prompt or file path
    supported_file_extensions:
      - .pdf
      - .docx
      - .html
      - .md
      # ... more extensions
```

**LLM-Powered PDF Extraction:**
When `llm_ingestion: true`, PDFs are processed using an LLM for better extraction quality. You can customize the prompt by providing either:
- Direct prompt text
- Path to a prompt file (`.md`, `.txt`, `.prompt`)

### Summarization

Creates document summaries for better context understanding.

```yaml
pipeline:
  summarization:
    max_tokens: 32768
    token_overlap: 512
    encoding_name: cl100k_base
    summarization_user_prompt: ""  # Custom prompt or file path
    combine_summaries_user_prompt: ""  # Custom prompt or file path
```

### Chunking

Splits documents into manageable chunks for question generation.

```yaml
pipeline:
  chunking:
    l_max_tokens: 8192      # Maximum chunk size
    token_overlap: 512      # Overlap between chunks
    encoding_name: cl100k_base
    h_min: 2               # Minimum hop distance
    h_max: 5               # Maximum hop distance
    num_multihops_factor: 1  # Multi-hop generation factor
```

### Question Generation

YourBench supports three question generation strategies:

#### Single-Shot Question Generation

```yaml
pipeline:
  single_shot_question_generation:
    question_mode: open-ended  # or "multi-choice"
    additional_instructions: "Focus on technical details"
    single_shot_system_prompt: path/to/prompt.md
    single_shot_system_prompt_multi: path/to/multi_choice_prompt.md
    single_shot_user_prompt: path/to/user_prompt.md
```

#### Multi-Hop Question Generation

```yaml
pipeline:
  multi_hop_question_generation:
    question_mode: open-ended
    additional_instructions: ""
    multi_hop_system_prompt: path/to/prompt.md
    multi_hop_system_prompt_multi: path/to/multi_choice_prompt.md
    multi_hop_user_prompt: path/to/user_prompt.md
```

#### Cross-Document Question Generation

```yaml
pipeline:
  cross_document_question_generation:
    question_mode: open-ended
    max_combinations: 100
    chunks_per_document: 1
    num_docs_per_combination: [2, 5]  # [min, max]
    random_seed: 42
    # ... prompt configurations
```

### Question Rewriting

Improves question naturalness and clarity.

```yaml
pipeline:
  question_rewriting:
    additional_instructions: "Make questions conversational"
    question_rewriting_system_prompt: path/to/prompt.md
    question_rewriting_user_prompt: path/to/prompt.md
```

### LightEval

Prepares data for evaluation.

```yaml
pipeline:
  prepare_lighteval:
    run: true
  
  lighteval:
    run: true
```

### Citation Score Filtering

Filters questions based on citation quality metrics.

```yaml
pipeline:
  citation_score_filtering:
    subset: prepared_lighteval
    alpha: 0.7  # Weight for metric 1
    beta: 0.3   # Weight for metric 2 (alpha + beta = 1.0)
```

## Advanced Features

### Environment Variables

YourBench automatically expands environment variables prefixed with `$`:

```yaml
hf_configuration:
  hf_token: $HF_TOKEN
  hf_organization: $HF_ORGANIZATION

model_list:
  - api_key: $OPENAI_API_KEY
```

**Special Behavior:**
- If `$HF_ORGANIZATION` is not set, YourBench attempts to retrieve it from the Hugging Face token

### Custom Prompts

Prompts can be specified in three ways:

1. **Built-in defaults** - Leave field empty
2. **Inline text** - Provide prompt directly in YAML
3. **File reference** - Path to `.md`, `.txt`, or `.prompt` file

```yaml
# Method 1: Use default
summarization_user_prompt: ""

# Method 2: Inline
summarization_user_prompt: "Summarize the key points..."

# Method 3: File reference
summarization_user_prompt: path/to/custom_prompt.md
```

**Package Resource Loading:**
YourBench first attempts to load prompts from the installed package's `yourbench.prompts` directory, ensuring compatibility with pip-installed versions.

### Model Role Assignment

Assign specific models to pipeline stages:

```yaml
model_list:
  - model_name: gpt-4o
  - model_name: claude-3-opus

model_roles:
  ingestion: [gpt-4o]
  summarization: [claude-3-opus]
  question_generation: [gpt-4o]
  # Unspecified stages use the first model
```

## Configuration Examples

### Minimal Configuration

```yaml
hf_configuration:
  hf_dataset_name: my-benchmark

model_list:
  - model_name: gpt-4o

pipeline:
  ingestion:
    source_documents_dir: data/documents
  single_shot_question_generation:
```

### Advanced Configuration

```yaml
hf_configuration:
  hf_dataset_name: advanced-benchmark
  hf_organization: my-org
  private: true
  concat_if_exist: true

model_list:
  - model_name: gpt-4o
    base_url: https://api.openai.com/v1/
    api_key: $OPENAI_API_KEY
    max_concurrent_requests: 50
  
  - model_name: claude-3-opus
    base_url: https://api.anthropic.com/v1/
    api_key: $ANTHROPIC_API_KEY

model_roles:
  ingestion: [gpt-4o]
  summarization: [claude-3-opus]
  question_generation: [gpt-4o]

pipeline:
  ingestion:
    source_documents_dir: data/raw
    output_dir: data/processed
    llm_ingestion: true
    pdf_llm_prompt: prompts/pdf_extraction.md
  
  summarization:
    max_tokens: 64000
    summarization_user_prompt: prompts/custom_summary.md
  
  chunking:
    l_max_tokens: 16384
    h_min: 3
    h_max: 7
  
  single_shot_question_generation:
    question_mode: multi-choice
    additional_instructions: "Create challenging technical questions"
    single_shot_system_prompt_multi: prompts/mc_system.md
  
  multi_hop_question_generation:
    question_mode: open-ended
  
  question_rewriting:
    additional_instructions: "Ensure questions are clear and unambiguous"
  
  prepare_lighteval:
  
  citation_score_filtering:
    alpha: 0.8
    beta: 0.2
```

## API Reference

### Loading Configuration

```python
from yourbench.utils.configuration_engine import YourbenchConfig

# Load from YAML file
config = YourbenchConfig.from_yaml("config.yaml")

# Create programmatically
config = YourbenchConfig(
    hf_configuration=HuggingFaceConfig(
        hf_dataset_name="my-dataset"
    ),
    model_list=[
        ModelConfig(model_name="gpt-4o")
    ]
)
```

### Accessing Configuration

```python
# Check if stage is enabled
if config.is_stage_enabled("ingestion"):
    ingestion_config = config.pipeline_config.ingestion
    
# Get model for stage
model_name = config.get_model_for_stage("summarization")

# Get enabled stages in execution order
enabled_stages = config.pipeline_config.get_enabled_stages()
```

### Saving Configuration

```python
# Save to YAML file
config.to_yaml("output_config.yaml")

# Get YAML string
yaml_str = config.model_dump_yaml()
```

### Validation

All configuration is validated using Pydantic:

- Type checking
- Range validation (e.g., DPI between 72-600)
- Constraint validation (e.g., alpha + beta = 1.0)
- Path validation and auto-creation
- Environment variable expansion

## Best Practices

1. **Use environment variables** for sensitive data like API keys
2. **Start with minimal config** and add stages as needed
3. **Leverage default prompts** before writing custom ones
4. **Test with small datasets** before processing large corpora
5. **Enable debug mode** for troubleshooting: `debug: true`
6. **Use model roles** when different stages need different models
7. **Version control** your configuration files
8. **Document custom prompts** clearly for team collaboration

## Troubleshooting

Common issues and solutions:

- **Missing environment variables**: Export required vars or use `.env` file
- **Path not found**: Use absolute paths or paths relative to config file
- **Validation errors**: Check field types and constraints in error message
- **Model assignment**: Ensure model names in `model_roles` match `model_list`
- **Prompt loading**: Verify file paths and extensions for custom prompts

For more examples, see the `example/` directory in the YourBench repository.
