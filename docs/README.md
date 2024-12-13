# Yourbench Documentation

Yourbench is a powerful tool for dynamically generating evaluation sets from seed documents, addressing the limitations of static benchmarks and benchmark saturation. This documentation provides a comprehensive guide to understanding and using Yourbench.

## Table of Contents
1. [Overview](#overview)
2. [Configuration Parameters](#configuration-parameters)
3. [Pipeline Components](#pipeline-components)
4. [Usage](#usage)

## Overview

Yourbench allows you to:
- Generate evaluation datasets from source documents
- Create semantic chunks from documents
- Generate both single-shot and multi-hop questions
- Push datasets to Hugging Face
- Customize the evaluation process for different test audiences

## Configuration Parameters

The configuration file (`config.yaml`) contains several key sections:

### Basic Configuration
- `task_name`: Identifier for your task, used in logs and output files

### Hugging Face Integration
- `push_to_huggingface`: Boolean flag to control dataset publishing to Hugging Face
- `set_hf_repo_visibility`: Sets repository visibility ("public" or "private")
- `hf_organization`: Target organization for dataset publishing

### Model Configuration
- `model_name`: Name/path of the model to use (e.g., "meta-llama/Llama-3.3-70B-Instruct")
- `model_type`: Type of model interface ("openai", "azure")
- `max_concurrent_requests`: Maximum number of concurrent API requests

### Pipeline Components

#### 1. Dataset Generation
```yaml
generate_dataset:
  execute: boolean
  files_directory: path/to/source/docs
  dataset_name: output_dataset_name
```

#### 2. Summary Generation
```yaml
generate_summaries:
  execute: boolean
  document_dataset_name: input_dataset
  summary_dataset_name: output_dataset
```

#### 3. Chunk Creation
```yaml
create_chunks:
  execute: boolean
  source_dataset_name: input_dataset
  chunked_documents_dataset_name: output_dataset
  chunking_configuration:
    model_name: embedding_model_name
    min_tokens: minimum_chunk_size
    target_chunk_size: desired_chunk_size
    max_tokens: maximum_chunk_size
    similarity_threshold: threshold_for_merging
    device: cuda/cpu
```

#### 4. Multi-hop Chunk Creation
```yaml
make_multihop_chunks:
  execute: boolean
  source_dataset_name: input_dataset
  multihop_pairings_dataset_name: output_dataset
```

#### 5. Single-Shot Question Generation
```yaml
create_single_shot_questions:
  execute: boolean
  source_dataset_name: input_dataset
  single_shot_questions_dataset_name: output_dataset
  prompt_prefix: prompt_template_name
  test_audience: target_audience_description
```

#### 6. Multi-hop Question Generation
```yaml
create_multihop_questions:
  execute: boolean
  source_dataset_name: input_dataset
  multihop_questions_dataset_name: output_dataset
  prompt_prefix: prompt_template_name
  test_audience: target_audience_description
```

## Pipeline Components Details

### 1. Dataset Generation
- Processes source documents from the specified directory
- Creates initial dataset structure

### 2. Summary Generation
- Generates summaries for input documents
- Uses specified model for summarization

### 3. Chunk Creation
- Creates semantic chunks from documents
- Uses embedding model for semantic similarity
- Configurable chunk sizes and merging thresholds

### 4. Multi-hop Chunks
- Creates pairings of chunks for multi-hop questions
- Enables complex reasoning across document sections

### 5. Question Generation
- Supports both single-shot and multi-hop questions
- Customizable for different test audiences
- Uses prompt templates from the `prompts` directory