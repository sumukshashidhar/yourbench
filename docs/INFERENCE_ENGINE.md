# Inference Engine Documentation

## Overview

The **InferenceEngine** is a Python module designed to facilitate interactions with various Large Language Models (LLMs) such as OpenAI's GPT models, Azure OpenAI, Anthropic's Claude, and Google's Gemini. It provides a unified interface for performing single and parallel inferences, handling retries, logging, and managing hyperparameters.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started](#getting-started)
  - [Initialization](#initialization)
  - [Performing Inference](#performing-inference)
  - [Parallel Inference](#parallel-inference)
- [Classes and Methods](#classes-and-methods)
  - [Hyperparameters](#hyperparameters)
  - [InferenceEngine](#inferenceengine)
    - [`__init__`](#__init__)
    - [`single_inference`](#single_inference)
    - [`parallel_inference`](#parallel_inference)
- [Logging](#logging)
- [Retry Mechanism](#retry-mechanism)
- [Testing](#testing)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Unified Interface**: Interact with multiple LLM providers using a consistent API.
- **Asynchronous Support**: Efficiently handle multiple inference requests in parallel.
- **Retry Mechanism**: Robust retry logic with exponential backoff for handling transient errors.
- **Logging**: Comprehensive logging for both console and file outputs, including inference logs.
- **Hyperparameter Management**: Easily customize inference hyperparameters.
- **Extensibility**: Designed to be extended for additional strategies and models.

## Prerequisites

- Python 3.7 or higher
- API keys and endpoints for the LLM providers you intend to use (e.g., OpenAI, Azure OpenAI, Anthropic, Gemini)

## Getting Started

### Initialization

To start using the InferenceEngine, you'll need to initialize it with the appropriate connection details and specify the model you wish to use.

```python
from inference_engine import InferenceEngine

connection_details = {
    "strategy": "openai",  # Options: "openai", "azure", "anthropic", "gemini"
    "api_key": "your_api_key",
    "base_url": "https://api.openai.com/v1",
}

engine = InferenceEngine(connection_details, model_name="gpt-3.5-turbo")
```

### Performing Inference

#### Single Inference

You can perform a single inference by calling the `single_inference` method with a list of message dictionaries.

```python
messages = [
    {"role": "user", "content": "Hello, how are you?"}
]

result, usage = engine.single_inference(messages)
print("Result:", result)
print("Usage:", usage)
```

#### Parallel Inference

For performing multiple inferences in parallel, use the `parallel_inference` method.

```python
messages_list = [
    [{"role": "user", "content": "Tell me a joke."}],
    [{"role": "user", "content": "What's the weather like today?"}],
    # Add more message lists
]

results, usages = engine.parallel_inference(messages_list)
for result in results:
    print("Result:", result)
```

### Parallel Inference

The `parallel_inference` method efficiently handles multiple inference requests asynchronously.

```python
results, usages = engine.parallel_inference(
    messages=messages_list,
    hyperparameters=Hyperparameters(temperature=0.7),
    max_concurrent_requests=10
)
```

## Classes and Methods

### Hyperparameters

A class for managing inference hyperparameters.

#### Initialization

```python
hyperparams = Hyperparameters(temperature=0.5, max_tokens=150)
```

#### Attributes

- `kwargs`: A dictionary of hyperparameter key-value pairs.

### InferenceEngine

A class that handles inference requests to LLMs.

#### `__init__`

Initialize the InferenceEngine.

```python
engine = InferenceEngine(connection_details, model_name="model_name")
```

- `connection_details` (dict): Contains strategy and API credentials.
- `model_name` (str): The name of the model to use.

#### `single_inference`

Perform a single inference request.

```python
result, usage = engine.single_inference(messages, hyperparameters)
```

- `messages` (list): A list of message dictionaries.
- `hyperparameters` (Hyperparameters): (Optional) Inference hyperparameters.

Returns:

- `result` (str): The generated response.
- `usage` (dict): Token usage statistics.

#### `parallel_inference`

Perform multiple inference requests in parallel.

```python
results, usages = engine.parallel_inference(
    messages_list,
    hyperparameters,
    max_concurrent_requests=10
)
```

- `messages_list` (list of lists): Each sublist contains message dictionaries for one inference.
- `hyperparameters` (Hyperparameters): (Optional) Inference hyperparameters.
- `max_concurrent_requests` (int): (Optional) Max number of concurrent requests.

Returns:

- `results` (list of str): Generated responses.
- `usages` (list of dict): Token usage statistics for each inference.

## Logging

The module uses Python's built-in `logging` library along with `loguru` for enhanced logging capabilities.

### Loggers

- **Default Logger**: Logs general information to both console and `logs/default.log`.
- **Inference Logger**: Logs detailed inference information to `logs/inference.log`.

### Log Format

Logs include timestamps, log levels, and messages. Console logs are color-coded for better readability.

### Custom Logging Decorators

- `@log_inference`: Decorator for logging synchronous inference methods.
- `@async_log_inference`: Decorator for logging asynchronous inference methods.

## Retry Mechanism

The module incorporates robust retry logic using the `tenacity` library.

### Configuration

Environment variables control the retry behavior:

- `RETRY_ATTEMPTS`: Number of retry attempts (default: 10)
- `RETRY_WAIT_EXPONENTIAL_MULTIPLIER`: Multiplier for exponential backoff (default: 1)
- `RETRY_WAIT_EXPONENTIAL_MIN`: Minimum wait time in seconds (default: 4)
- `RETRY_WAIT_EXPONENTIAL_MAX`: Maximum wait time in seconds (default: 300)

### Usage

The `@retry` decorator is applied to methods that make external API calls to handle transient errors gracefully.

## Testing

The module includes a suite of unit tests using Python's `unittest` framework.

### Running Tests

To run the tests, execute:

```bash
python inference_engine.py
```

### Test Cases

- **Strategy Compatibility Tests**: Ensure that only compatible strategies are accepted.
- **Single Inference Tests**: Test the `single_inference` method with valid inputs.
- **Parallel Inference Tests**: Test the `parallel_inference` method with multiple messages.
- **Async Inference Tests**: Test asynchronous inference methods for different strategies.

### Environment Variable for Tests

- `RUN_EXPENSIVE_TESTS`: Set to `true` to run tests that make actual API calls.

## Examples

### Example 1: Basic Single Inference

```python
from inference_engine import InferenceEngine, Hyperparameters

connection_details = {
    "strategy": "openai",
    "api_key": "your_openai_api_key",
    "base_url": "https://api.openai.com/v1",
}

engine = InferenceEngine(connection_details, model_name="gpt-3.5-turbo")
messages = [{"role": "user", "content": "Explain the theory of relativity."}]
hyperparams = Hyperparameters(temperature=0.7)

result, usage = engine.single_inference(messages, hyperparams)
print("AI Response:", result)
```

### Example 2: Parallel Inference with Custom Hyperparameters

```python
from inference_engine import InferenceEngine, Hyperparameters

connection_details = {
    "strategy": "azure",
    "api_key": "your_azure_api_key",
    "azure_endpoint": "https://your_azure_endpoint/",
    "api_version": "2023-03-15-preview",
}

engine = InferenceEngine(connection_details, model_name="gpt-4")
messages_list = [
    [{"role": "user", "content": "Tell me about quantum computing."}],
    [{"role": "user", "content": "What is the capital of France?"}],
    # Add more messages as needed
]

hyperparams = Hyperparameters(temperature=0.5, max_tokens=100)

results, usages = engine.parallel_inference(messages_list, hyperparams)
for idx, result in enumerate(results):
    print(f"Response {idx+1}:", result)
```

### Example 3: Handling Different Strategies

```python
from inference_engine import InferenceEngine

# For Anthropic's Claude
connection_details = {
    "strategy": "anthropic",
    "api_key": "your_anthropic_api_key",
}

engine = InferenceEngine(connection_details, model_name="claude-v1")
messages = [{"role": "user", "content": "Summarize the plot of '1984' by George Orwell."}]

result, usage = engine.single_inference(messages)
print("AI Response:", result)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

### Adding New Strategies

To add support for a new LLM provider:

1. Implement the client initialization in `_initialize_client`.
2. Implement the inference methods:
   - `_async_single_message_inference`
   - Synchronous counterparts if necessary.
3. Update the `COMPATIBLE_STRATEGIES` list.
4. Add test cases for the new strategy.

## License

This project is licensed under the MIT License.

---

Feel free to customize and extend the InferenceEngine to suit your specific needs. For any questions or support, please contact the maintainers.