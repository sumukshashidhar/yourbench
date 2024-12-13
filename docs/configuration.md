# Yourbench Configuration Guide

This document provides a comprehensive guide to configuring Yourbench through its YAML configuration file.

## Configuration Structure

The configuration file is divided into several main sections:

1. Basic Task Configuration
2. Hugging Face Integration
3. Model Configuration
4. Pipeline Components

## Basic Task Configuration

```yaml
task_name: yourbench_y1  # Unique identifier for your task
```

- `task_name`: Used to identify the task in logs and output files. This name is also used to locate the task's configuration directory.

## Hugging Face Integration

```yaml
configurations:
  push_to_huggingface: true
  set_hf_repo_visibility: public
  hf_organization: your-org-name
```

- `push_to_huggingface`: Boolean flag to control whether datasets are pushed to Hugging Face Hub
- `set_hf_repo_visibility`: Sets the visibility of the Hugging Face repository ("public" or "private")
- `hf_organization`: Target organization on Hugging Face where datasets will be pushed

## Model Configuration

```yaml
configurations:
  model:
    model_name: model-name/or-path
    model_type: openai|azure
    max_concurrent_requests: 512
```

- `model_name`: Name or path of the model to use (e.g., "meta-llama/Llama-3.3-70B-Instruct")
- `model_type`: Type of model interface:
  - `openai`: Uses OpenAI-compatible API
  - `azure`: Uses Azure OpenAI API
- `max_concurrent_requests`: Maximum number of concurrent API requests for parallel processing

## Pipeline Components

### 1. Dataset Generation

```yaml
generate_dataset:
  execute: boolean
  files_directory: path/to/source/docs
  dataset_name: output_dataset_name
```

- `execute`: Controls whether this pipeline component runs
- `files_directory`: Directory containing source documents
- `dataset_name`: Name for the generated dataset

### 2. Summary Generation

```yaml
generate_summaries:
  execute: boolean
  document_dataset_name: input_dataset
  summary_dataset_name: output_dataset
```

- `execute`: Controls whether this pipeline component runs
- `document_dataset_name`: Name of the input dataset containing documents
- `summary_dataset_name`: Name for the output dataset containing summaries

### 3. Chunk Creation

```yaml
create_chunks:
  execute: boolean
  source_dataset_name: input_dataset
  chunked_documents_dataset_name: output_dataset
  chunking_configuration:
    model_name: sentence-transformers/model-name
    min_tokens: 256
    target_chunk_size: 512
    max_tokens: 1024
    similarity_threshold: 0.3
    device: cuda|cpu
```

- `execute`: Controls whether this pipeline component runs
- `source_dataset_name`: Name of the input dataset
- `chunked_documents_dataset_name`: Name for the output chunked dataset
- Chunking Configuration:
  - `model_name`: Sentence transformer model for semantic similarity
  - `min_tokens`: Minimum chunk size in tokens
  - `target_chunk_size`: Desired chunk size in tokens
  - `max_tokens`: Maximum chunk size in tokens
  - `similarity_threshold`: Threshold for merging chunks (0-1)
  - `device`: Computing device ("cuda" for GPU, "cpu" for CPU)

### 4. Multi-hop Chunk Creation

```yaml
make_multihop_chunks:
  execute: boolean
  source_dataset_name: input_dataset
  multihop_pairings_dataset_name: output_dataset
```

- `execute`: Controls whether this pipeline component runs
- `source_dataset_name`: Name of the input chunked dataset
- `multihop_pairings_dataset_name`: Name for the output dataset containing multi-hop chunk pairings

### 5. Single-Shot Question Generation

```yaml
create_single_shot_questions:
  execute: boolean
  source_dataset_name: input_dataset
  single_shot_questions_dataset_name: output_dataset
  prompt_prefix: prompt_template_name
  test_audience: audience_description
```

- `execute`: Controls whether this pipeline component runs
- `source_dataset_name`: Name of the input chunked dataset
- `single_shot_questions_dataset_name`: Name for the output questions dataset
- `prompt_prefix`: Name of the prompt template to use (e.g., "simple_qg")
- `test_audience`: Description of the target audience (affects question complexity)

### 6. Multi-hop Question Generation

```yaml
create_multihop_questions:
  execute: boolean
  source_dataset_name: input_dataset
  multihop_questions_dataset_name: output_dataset
  prompt_prefix: prompt_template_name
  test_audience: audience_description
```

- `execute`: Controls whether this pipeline component runs
- `source_dataset_name`: Name of the input multi-hop dataset
- `multihop_questions_dataset_name`: Name for the output questions dataset
- `prompt_prefix`: Name of the prompt template to use (e.g., "simple_qg")
- `test_audience`: Description of the target audience (affects question complexity)

## Environment Variables

The following environment variables are required for model access:

For OpenAI:
- `MODEL_BASE_URL`: Base URL for OpenAI API
- `MODEL_API_KEY`: OpenAI API key

For Azure:
- `AZURE_BASE_URL`: Base URL for Azure OpenAI API
- `AZURE_API_KEY`: Azure OpenAI API key

## Question Types

The system supports various question types for both single-shot and multi-hop questions:

1. `analytical`: Questions that break down complex ideas or relationships
2. `application-based`: Questions that apply concepts to new scenarios
3. `clarification`: Questions seeking deeper understanding of specific points
4. `counterfactual`: Questions exploring alternative scenarios
5. `conceptual`: Questions examining key terms and theories
6. `true-false`: Questions verifying understanding with boolean statements
7. `factual`: Questions testing recall of explicit information
8. `open-ended`: Questions encouraging broader discussion
9. `false-premise`: Questions correcting misconceptions
10. `edge-case`: Questions testing boundary conditions

## Pipeline Flow

1. **Dataset Generation**: Creates initial dataset from source documents
2. **Summary Generation**: Generates summaries for input documents
3. **Chunk Creation**: Creates semantic chunks using embedding model
4. **Multi-hop Chunk Creation**: Creates pairings for multi-hop questions
5. **Question Generation**: Generates both single-shot and multi-hop questions

Each step can be enabled/disabled using the `execute` flag in the configuration.