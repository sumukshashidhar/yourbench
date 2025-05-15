# YourBench Configuration Documentation

## Configuration File Overview

The YourBench configuration file is written in YAML and consists of several key sections, each controlling a distinct part of the tool's functionality:

- **`settings`**: Global settings that apply across the entire application.
- **`hf_configuration`**: Settings for integration with the Hugging Face Hub.
- **`local_dataset_dir`**: Optional path for local dataset storage instead of Hugging Face Hub.
- **`model_list`**: Definitions of the models available for use in YourBench.
- **`model_roles`**: Assignments of models to specific pipeline stages.
- **`pipeline`**: Configuration for the stages of the YourBench pipeline.

Below, each section is detailed with descriptions, YAML syntax, and examples to help you configure YourBench effectively.

---

## Global Settings

The `settings` section defines global options that influence the overall behavior of YourBench.

### YAML Syntax
```yaml
settings:
  debug: false  # Enable debug mode with metrics collection
```

### Options
- **`debug`**  
  - **Type**: Boolean  
  - **Default**: `false`  
  - **Description**: When set to `true`, enables debug mode, which collects additional metrics during execution for troubleshooting or analysis. In debug mode, perplexity and readability metrics are calculated for chunks, and similarity plots may be generated.

### Example
To enable debug mode:
```yaml
settings:
  debug: true
```

---

## Hugging Face Settings

The `hf_configuration` section manages integration with the Hugging Face Hub, including authentication and dataset handling.

### YAML Syntax
```yaml
hf_configuration:
  token: $HF_TOKEN  # Hugging Face API token
  hf_organization: $HF_ORGANIZATION  # Optional: Organization name
  private: false  # Dataset visibility
  hf_dataset_name: yourbench_example  # Dataset name for traces and questions
  concat_if_exist: false  # Whether to concatenate with existing dataset
```

### Options
- **`token`**  
  - **Type**: String  
  - **Required**: Yes  
  - **Description**: Your Hugging Face API token, obtainable from [Hugging Face's documentation](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication). Use an environment variable (e.g., `$HF_TOKEN`) for security.

- **`hf_organization`**  
  - **Type**: String  
  - **Optional**: Yes  
  - **Default**: Your Hugging Face username  
  - **Description**: Specifies the organization under which datasets are created. If omitted, defaults to your username.

- **`private`**  
  - **Type**: Boolean  
  - **Default**: `true`  
  - **Description**: Controls dataset visibility on the Hugging Face Hub. Set to `false` for public datasets, `true` for private ones.

- **`hf_dataset_name`**  
  - **Type**: String  
  - **Required**: Yes  
  - **Description**: The name of the dataset on the Hugging Face Hub where traces and generated questions are stored.

- **`concat_if_exist`**  
  - **Type**: Boolean  
  - **Default**: `false`  
  - **Description**: If set to `true`, new data will be concatenated to an existing dataset with the same name. If `false`, the existing dataset will be overwritten.

### Example
For a public dataset under a specific organization:
```yaml
hf_configuration:
  token: $HF_TOKEN
  hf_organization: my_org
  private: false
  hf_dataset_name: yourbench_data
  concat_if_exist: false
```

---

## Local Dataset Directory

As an alternative to storing datasets on the Hugging Face Hub, you can specify a local directory for dataset storage.

### YAML Syntax
```yaml
local_dataset_dir: /path/to/local/dataset
```

### Options
- **`local_dataset_dir`**  
  - **Type**: String  
  - **Optional**: Yes  
  - **Description**: Path to a local directory where datasets will be stored. If specified, this overrides the Hugging Face Hub storage.

### Example
```yaml
local_dataset_dir: /home/user/yourbench_datasets
```

---

## Model Settings

The `model_list` section defines the models YourBench can utilize, supporting various providers and inference backends.

### YAML Syntax
```yaml
model_list:
  # Hugging Face model
  - model_name: Qwen/Qwen2.5-VL-72B-Instruct
    provider: hf-inference  # Optional: e.g., hf-inference, novita, together
    api_key: $HF_TOKEN
    max_concurrent_requests: 32
    base_url: null  # Optional: custom API endpoint

  # OpenAI model
  - model_name: gpt-4o
    provider: null  # Set to null for OpenAI API
    base_url: "https://api.openai.com/v1"  # Default OpenAI API URL
    api_key: $OPENAI_API_KEY
    max_concurrent_requests: 10

  # OpenAI-compatible model (e.g., local server, Azure, Anthropic)
  - model_name: claude-3-opus-20240229
    provider: null
    base_url: "https://api.anthropic.com/v1"  # Replace with your API endpoint
    api_key: $ANTHROPIC_API_KEY
    max_concurrent_requests: 5
```

### Options
For each model in the list, you can specify:

- **`model_name`**  
  - **Type**: String  
  - **Required**: Yes  
  - **Description**: The identifier or name of the model (e.g., `gpt-4o`, `Qwen/Qwen2.5-VL-72B-Instruct`).

- **`provider`**  
  - **Type**: String  
  - **Optional**: Yes  
  - **Default**: `null`  
  - **Description**: For Hugging Face models, specifies the inference provider (e.g., `hf-inference`, `novita`, `together`). Set to `null` for OpenAI API or OpenAI-compatible endpoints.

- **`api_key`**  
  - **Type**: String  
  - **Required**: Yes (for most models)  
  - **Description**: The API key for accessing the model, typically stored as an environment variable.

- **`base_url`**  
  - **Type**: String  
  - **Optional**: Yes  
  - **Description**: The base URL for the API endpoint. Use this to specify custom endpoints for OpenAI-compatible APIs or local inference servers.

- **`max_concurrent_requests`**  
  - **Type**: Integer  
  - **Optional**: Yes  
  - **Default**: `16`  
  - **Description**: Limits the number of concurrent requests to the model.

### Example
Configuring a mix of remote and local models:
```yaml
model_list:
  - model_name: gpt-4o
    provider: null
    base_url: "https://api.openai.com/v1"
    api_key: $OPENAI_API_KEY
    max_concurrent_requests: 10
  
  - model_name: local-llama-3
    provider: null
    base_url: "http://localhost:8000/v1"
    api_key: null
    max_concurrent_requests: 1
```

---

## Model Roles

The `model_roles` section assigns models from `model_list` to specific stages of the YourBench pipeline. Each stage can use one or multiple models.

### YAML Syntax
```yaml
model_roles:
  ingestion:
    - Qwen/Qwen2.5-VL-72B-Instruct  # Vision-supported model required for multimodality
  summarization:
    - Qwen/Qwen2.5-72B-Instruct
  chunking:
    - intfloat/multilingual-e5-large-instruct  # Embedding model for semantic chunking
  single_shot_question_generation:
    - Qwen/Qwen2.5-72B-Instruct
  multi_hop_question_generation:
    - Qwen/Qwen2.5-72B-Instruct
```

### Available Roles
- **`ingestion`**  
  - **Description**: Models used for document ingestion and conversion to markdown. Vision-supported models are required for processing images.
  - **Recommended**: Vision-capable models like `Qwen/Qwen2.5-VL-72B-Instruct`, `gpt-4o`, etc.

- **`summarization`**  
  - **Description**: Models used to generate document summaries.
  - **Recommended**: Strong language models with good summarization capabilities.

- **`chunking`**  
  - **Description**: Models used for semantic chunking (embedding models).
  - **Recommended**: Sentence embedding models like `intfloat/multilingual-e5-large-instruct`.

- **`single_shot_question_generation`**  
  - **Description**: Models used to generate questions from individual chunks.
  - **Recommended**: Strong language models with good question generation capabilities.

- **`multi_hop_question_generation`**  
  - **Description**: Models used to generate questions that require reasoning across multiple chunks.
  - **Recommended**: Strong language models with good reasoning capabilities.

### Notes
- For the `ingestion` stage, a vision-supported model is required if your documents contain images.
- Multiple models per stage allow for flexibility or experimentation.
- If no model is specified for a role, the first model in `model_list` will be used.

### Example
```yaml
model_roles:
  ingestion:
    - gpt-4o
  summarization:
    - claude-3-opus-20240229
  chunking:
    - intfloat/multilingual-e5-large-instruct
  single_shot_question_generation:
    - gpt-4o
  multi_hop_question_generation:
    - claude-3-opus-20240229
```

---

## Pipeline Stages

The `pipeline` section configures the stages of the YourBench workflow. Each stage can be enabled or disabled and includes additional settings specific to its functionality.

### Common Options for All Stages
- **`run`**  
  - **Type**: Boolean  
  - **Required**: Yes  
  - **Description**: Enables (`true`) or disables (`false`) the stage.

### Ingestion Stage

The ingestion stage converts source documents from various formats to markdown.

#### YAML Syntax
```yaml
pipeline:
  ingestion:
    run: true
    source_documents_dir: example/data/raw
    output_dir: example/data/processed
```

#### Options
- **`source_documents_dir`**  
  - **Type**: String  
  - **Required**: Yes  
  - **Description**: Directory containing the raw source documents to be processed.

- **`output_dir`**  
  - **Type**: String  
  - **Required**: Yes  
  - **Description**: Directory where the converted markdown files will be saved.

### Upload Ingest to Hub Stage

This stage uploads the processed documents to the Hugging Face Hub or local dataset.

#### YAML Syntax
```yaml
pipeline:
  upload_ingest_to_hub:
    run: true
    source_documents_dir: example/data/processed
```

#### Options
- **`source_documents_dir`**  
  - **Type**: String  
  - **Required**: Yes  
  - **Description**: Directory containing the processed markdown files to be uploaded.

### Summarization Stage

This stage generates summaries for each document.

#### YAML Syntax
```yaml
pipeline:
  summarization:
    run: true
```

#### Options
- No additional options required. The stage uses the dataset from the previous stage.

### Chunking Stage

This stage splits documents into chunks for question generation.

#### YAML Syntax
```yaml
pipeline:
  chunking:
    run: true
    chunking_configuration:
      l_min_tokens: 64
      l_max_tokens: 128
      tau_threshold: 0.8
      h_min: 2
      h_max: 5
      num_multihops_factor: 2
      chunking_mode: "fast_chunking"
```

#### Options
- **`chunking_configuration`**  
  - **Type**: Object  
  - **Optional**: Yes  
  - **Description**: Configuration for the chunking process.

  - **`l_min_tokens`**  
    - **Type**: Integer  
    - **Default**: `64`  
    - **Description**: Minimum number of tokens in each chunk.

  - **`l_max_tokens`**  
    - **Type**: Integer  
    - **Default**: `128`  
    - **Description**: Maximum number of tokens in each chunk.

  - **`tau_threshold`**  
    - **Type**: Float  
    - **Default**: `0.3`  
    - **Description**: Similarity threshold for determining chunk boundaries in semantic chunking.

  - **`h_min`**  
    - **Type**: Integer  
    - **Default**: `2`  
    - **Description**: Minimum number of unique chunks to combine for multi-hop questions.

  - **`h_max`**  
    - **Type**: Integer  
    - **Default**: `3`  
    - **Description**: Maximum number of unique chunks to combine for multi-hop questions.

  - **`num_multihops_factor`**  
    - **Type**: Integer or Float  
    - **Default**: `2`  
    - **Description**: Factor determining how many multi-hop combinations to generate. Higher values generate more combinations.

  - **`chunking_mode`**  
    - **Type**: String  
    - **Default**: `"fast_chunking"`  
    - **Options**: `"fast_chunking"` or `"semantic_chunking"`  
    - **Description**: Mode for chunking. `"fast_chunking"` uses purely length-based rules, while `"semantic_chunking"` uses sentence embeddings and similarity thresholds.

### Single Shot Question Generation Stage

This stage generates questions from individual chunks.

#### YAML Syntax
```yaml
pipeline:
  single_shot_question_generation:
    run: true
    additional_instructions: "Generate questions to test a curious adult"
    chunk_sampling:
      mode: "count"
      value: 5
      random_seed: 123
```

#### Options
- **`additional_instructions`**  
  - **Type**: String  
  - **Default**: `"Generate questions to test an undergraduate student"`  
  - **Description**: Additional instructions for the question generation model.

- **`chunk_sampling`**  
  - **Type**: Object  
  - **Optional**: Yes  
  - **Description**: Configuration for sampling chunks to reduce cost.

  - **`mode`**  
    - **Type**: String  
    - **Default**: `"all"`  
    - **Options**: `"all"`, `"count"`, or `"percentage"`  
    - **Description**: Mode for sampling chunks. `"all"` uses all chunks, `"count"` uses a fixed number, and `"percentage"` uses a percentage of chunks.

  - **`value`**  
    - **Type**: Integer or Float  
    - **Default**: `1.0`  
    - **Description**: Value for sampling. For `"count"` mode, this is the number of chunks. For `"percentage"` mode, this is the percentage (0.0-1.0) of chunks.

  - **`random_seed`**  
    - **Type**: Integer  
    - **Default**: `42`  
    - **Description**: Random seed for reproducible sampling.

### Multi-Hop Question Generation Stage

This stage generates questions that require reasoning across multiple chunks.

#### YAML Syntax
```yaml
pipeline:
  multi_hop_question_generation:
    run: true
    additional_instructions: "Generate questions to test a curious adult"
    chunk_sampling:
      mode: "percentage"
      value: 0.3
      random_seed: 42
```

#### Options
- **`additional_instructions`**  
  - **Type**: String  
  - **Default**: `"Generate questions to test an undergraduate student"`  
  - **Description**: Additional instructions for the question generation model.

- **`chunk_sampling`**  
  - **Type**: Object  
  - **Optional**: Yes  
  - **Description**: Configuration for sampling chunks to reduce cost.

  - **`mode`**  
    - **Type**: String  
    - **Default**: `"all"`  
    - **Options**: `"all"`, `"count"`, or `"percentage"`  
    - **Description**: Mode for sampling chunks. `"all"` uses all chunks, `"count"` uses a fixed number, and `"percentage"` uses a percentage of chunks.

  - **`value`**  
    - **Type**: Integer or Float  
    - **Default**: `1.0`  
    - **Description**: Value for sampling. For `"count"` mode, this is the number of chunks. For `"percentage"` mode, this is the percentage (0.0-1.0) of chunks.

  - **`random_seed`**  
    - **Type**: Integer  
    - **Default**: `42`  
    - **Description**: Random seed for reproducible sampling.

### LightEval Stage

This stage combines single-shot and multi-hop questions into a unified dataset for evaluation.

#### YAML Syntax
```yaml
pipeline:
  lighteval:
    run: true
```

#### Options
- No additional options required. The stage uses the datasets from the previous stages.

---

## Complete Configuration Example

```yaml
# === GLOBAL SETTINGS ===
settings:
  debug: false

# === HUGGINGFACE SETTINGS ===
hf_configuration:
  token: $HF_TOKEN
  hf_organization: $HF_ORGANIZATION
  private: false
  hf_dataset_name: yourbench_example
  concat_if_exist: false

# === MODEL CONFIGURATION ===
model_list: 
  - model_name: Qwen/Qwen2.5-VL-72B-Instruct
    provider: hf-inference
    api_key: $HF_TOKEN
    max_concurrent_requests: 32
  - model_name: Qwen/Qwen2.5-72B-Instruct
    provider: novita
    api_key: $HF_TOKEN
    max_concurrent_requests: 32

model_roles:
  ingestion:
    - Qwen/Qwen2.5-VL-72B-Instruct
  summarization:
    - Qwen/Qwen2.5-72B-Instruct
  chunking:
    - intfloat/multilingual-e5-large-instruct
  single_shot_question_generation:
    - Qwen/Qwen2.5-72B-Instruct
  multi_hop_question_generation:
    - Qwen/Qwen2.5-72B-Instruct

# === PIPELINE CONFIGURATION ===
pipeline:
  ingestion:
    run: true
    source_documents_dir: example/data/raw
    output_dir: example/data/processed

  upload_ingest_to_hub:
    run: true
    source_documents_dir: example/data/processed

  summarization:
    run: true
  
  chunking:
    run: true
    chunking_configuration:
      l_min_tokens: 64
      l_max_tokens: 128
      tau_threshold: 0.8
      h_min: 2
      h_max: 5
      num_multihops_factor: 2
      chunking_mode: "fast_chunking"
  
  single_shot_question_generation:
    run: true
    additional_instructions: "Generate questions to test a curious adult"
    chunk_sampling:
      mode: "count"
      value: 5
      random_seed: 123
  
  multi_hop_question_generation:
    run: true
    additional_instructions: "Generate questions to test a curious adult"
    chunk_sampling:
      mode: "percentage"
      value: 0.3
      random_seed: 42

  lighteval:
    run: true
```
