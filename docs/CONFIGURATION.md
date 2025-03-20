# YourBench Configuration Documentation

## Configuration File Overview

The YourBench configuration file is written in YAML and consists of several key sections, each controlling a distinct part of the tool's functionality:

- **`settings`**: Global settings that apply across the entire application.
- **`hf_configuration`**: Settings for integration with the Hugging Face Hub.
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
  - **Description**: When set to `true`, enables debug mode, which collects additional metrics during execution for troubleshooting or analysis.

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
  global_dataset_name: tempora_yourbench_traces  # Dataset name for traces and questions
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

- **`global_dataset_name`**  
  - **Type**: String  
  - **Required**: Yes  
  - **Description**: The name of the dataset on the Hugging Face Hub where traces and generated questions are stored.

### Example
For a public dataset under a specific organization:
```yaml
hf_configuration:
  token: $HF_TOKEN
  hf_organization: my_org
  private: false
  global_dataset_name: my_org/yourbench_data
```

---

## Model Settings

The `model_list` section defines the models YourBench can utilize, supporting various providers and inference backends. YourBench natively integrates with `litellm` and `hf_hub`, enabling compatibility with a wide range of models.

### YAML Syntax
```yaml
model_list:
  # Hugging Face model
  - model_name: Qwen/Qwen2-VL-72B-Instruct
    api_key: $HF_TOKEN
    provider: hyperbolic  # Optional: e.g., hyperbolic, together, novita
    inference_backend: hf_hub

  # OpenAI model
  - model_name: gpt-4o
    api_key: $OPENAI_API_KEY
    max_concurrent_requests: 512

  # Anthropic model
  - model_name: claude-3-7-sonnet-20250219
    request_style: anthropic
    api_key: $ANTHROPIC_API_KEY
    timeout: 600

  # Google model
  - model_name: gemini-2.0-flash-001
    request_style: google
    api_key: $GEMINI_API_KEY

  # DeepSeek model
  - model_name: deepseek-chat
    request_style: deepseek
    api_key: $DEEPSEEK_API_KEY

  # OpenRouter model
  - model_name: meta-llama/llama-3.1-405b-instruct
    request_style: openrouter
    api_key: $OPENROUTER_API_KEY

  # Local model (e.g., via vLLM)
  - model_name: YOUR_MODEL_NAME
    request_style: openai
    base_url: http://localhost:8000/v1
    max_concurrent_requests: 512
```

### Options
For each model in the list, you can specify:

- **`model_name`**  
  - **Type**: String  
  - **Required**: Yes  
  - **Description**: The identifier or name of the model (e.g., `gpt-4o`, `Qwen/Qwen2-VL-72B-Instruct`).

- **`api_key`**  
  - **Type**: String  
  - **Required**: Yes (for remote models)  
  - **Description**: The API key for accessing the model, typically stored as an environment variable.

- **`request_style`**  
  - **Type**: String  
  - **Default**: `openai`  
  - **Description**: The API style for the model (e.g., `openai`, `anthropic`, `google`, `deepseek`, `openrouter`).

- **`base_url`**  
  - **Type**: String  
  - **Optional**: Yes  
  - **Description**: The base URL for local inference servers (e.g., `http://localhost:8000/v1` for vLLM or SGLang).

- **`provider`**  
  - **Type**: String  
  - **Optional**: Yes  
  - **Description**: For Hugging Face models, specifies the inference provider (e.g., `hyperbolic`, `together`, `novita`). If omitted, routing is automatic.

- **`inference_backend`**  
  - **Type**: String  
  - **Default**: `litellm`  
  - **Description**: The backend for inference, either `litellm` or `hf_hub`.

- **`max_concurrent_requests`**  
  - **Type**: Integer  
  - **Optional**: Yes  
  - **Default**: 8  
  - **Description**: Limits the number of concurrent requests to the model.

- **`timeout`**  
  - **Type**: Integer  
  - **Optional**: Yes  
  - **Description**: Sets the timeout in seconds for model responses.

### Example
Configuring a mix of remote and local models:
```yaml
model_list:
  - model_name: gpt-4o
    api_key: $OPENAI_API_KEY
  - model_name: my_local_model
    request_style: openai
    base_url: http://localhost:8000/v1
    max_concurrent_requests: 256
```

---

## Model Roles

The `model_roles` section assigns models from `model_list` to specific stages of the YourBench pipeline. Each stage can use one or multiple models.

### YAML Syntax
```yaml
model_roles:
  ingestion:
    - Qwen/Qwen2-VL-72B-Instruct  # Vision-supported model required
  summarization:
    - deepseek-chat
  chunking:
    - intfloat/multilingual-e5-large-instruct
  single_shot_question_generation:
    - deepseek-chat
    - Qwen/Qwen2-VL-72B-Instruct
```

### Structure
- Each key under `model_roles` represents a pipeline stage (e.g., `ingestion`, `summarization`).
- The value is a list of model names (from `model_list`) assigned to that stage.

### Notes
- For the `ingestion` stage, a vision-supported model is required (e.g., `Qwen/Qwen2-VL-72B-Instruct`).
- Multiple models per stage allow for flexibility or experimentation.

### Example
Assigning models to different tasks:
```yaml
model_roles:
  ingestion:
    - Qwen/Qwen2-VL-72B-Instruct
  summarization:
    - gpt-4o
  chunking:
    - intfloat/multilingual-e5-large-instruct
```

---

## Pipeline Stages

The `pipeline` section configures the stages of the YourBench workflow. Each stage can be enabled or disabled and may include additional settings specific to its functionality.

### YAML Syntax (Placeholder)
```yaml
pipeline:
  ingestion:
    run: true
    source_documents_dir: data/example/raw
    output_dir: data/example/processed

  summarization:
    run: true
    source_subset: ingested_documents
    output_subset: summarized_documents

  chunking:
    run: true
    source_subset: summarized_documents
    output_subset: chunked_documents

  single_shot_question_generation:
    run: true
    source_subset: chunked_documents
    output_subset: single_shot_questions
```

### General Options
For each stage:
- **`run`**  
  - **Type**: Boolean  
  - **Required**: Yes  
  - **Description**: Enables (`true`) or disables (`false`) the stage.

- **Stage-Specific Options**: Each stage may include additional parameters (e.g., input/output directories, subsets). Refer to the YourBench implementation or further documentation for details.

### Example
A minimal pipeline configuration:
```yaml
pipeline:
  ingestion:
    run: true
    source_documents_dir: data/input
    output_dir: data/processed
  summarization:
    run: true
    source_subset: processed_documents
    output_subset: summaries
```

