# Using OpenAI Compatible Models

YourBench supports using any OpenAI-compatible model by configuring the `base_url` parameter in your YAML configuration.

## OpenRouter Example

OpenRouter exposes an OpenAI-compatible API. Set `base_url` and `OPENROUTER_API_KEY`. Provider-specific options (like `reasoning`) go in `extra_parameters`.

```yaml
model_list:
  - model_name: x-ai/grok-4-fast:free
    base_url: "https://openrouter.ai/api/v1"
    api_key: $OPENROUTER_API_KEY
    max_concurrent_requests: 16
    extra_parameters:
      reasoning:
        effort: medium
```

With uvx CLI:

```bash
export OPENROUTER_API_KEY=your_openrouter_key
uvx --from yourbench yourbench run example/default_example/config.yaml \
  --debug \
  --model x-ai/grok-4-fast:free \
  --base-url https://openrouter.ai/api/v1 \
  --model-extra-parameters '{"reasoning": {"effort": "medium"}}'
```

## Configuration

Add your OpenAI-compatible model to the `model_list` section of your configuration YAML:

```yaml
model_list:
  - model_name: gpt-4o
    base_url: "https://api.openai.com/v1"  # Default OpenAI API URL
    api_key: $OPENAI_API_KEY
    max_concurrent_requests: 10
    extra_parameters:
      reasoning:
        effort: medium

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

If your provider exposes additional request fields (for example OpenRouter's `reasoning` settings), set them in `extra_parameters` or supply them via `--model-extra-parameters` when using the CLI.

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
