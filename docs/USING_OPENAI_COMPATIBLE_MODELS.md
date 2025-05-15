# Using OpenAI Compatible Models

YourBench supports using any OpenAI-compatible model by configuring the `base_url` parameter in your YAML configuration.

## Configuration

Add your OpenAI-compatible model to the `model_list` section of your configuration YAML:

```yaml
model_list:
  - model_name: gpt-4o
    base_url: "https://api.openai.com/v1"  # Default OpenAI API URL
    api_key: $OPENAI_API_KEY
    max_concurrent_requests: 10

  # Example for an Anthropic Server
  - model_name: claude-3-7-sonnet-20250219
    provider: null
    base_url: "https://api.anthropic.com/v1/"  # Replace with your API endpoint
    api_key: $ANTHROPIC_API_KEY
    max_concurrent_requests: 5
```

## Environment Variables

Set the required API keys as environment variables. For example:

```bash
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Model Roles

Assign your models to specific pipeline roles:

```yaml
model_roles:
  ingestion:
    - gpt-4o  # For vision-supported tasks
  summarization:
    - claude-3-7-sonnet-20250219
  chunking:
    - intfloat/multilingual-e5-large-instruct
  single_shot_question_generation:
    - gpt-4o
  # using multiple models for question generation
  multi_hop_question_generation:
    - claude-3-7-sonnet-20250219
    - gpt-4o
```
