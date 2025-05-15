"""
Inference Engine For Yourbench - Now with true concurrency throttling and cost tracking.
"""

import os
import csv
import time
import uuid
import atexit
import asyncio
import datetime
import collections
from typing import Any, Dict, List, Optional
from dataclasses import field, dataclass

import tiktoken  # Added for token counting
from dotenv import load_dotenv
from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from huggingface_hub import AsyncInferenceClient


load_dotenv()

GLOBAL_TIMEOUT = 300

# Using defaultdict for easier accumulation
_cost_data = collections.defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0})
_individual_log_file = os.path.join("logs", "inference_cost_log_individual.csv")
_aggregate_log_file = os.path.join("logs", "inference_cost_log_aggregate.csv")
_individual_header_written = False


@dataclass
class Model:
    model_name: str
    # You can find the list of available providers here: https://huggingface.co/docs/huggingface_hub/guides/inference#supported-providers-and-tasks
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    bill_to: str | None = None
    max_concurrent_requests: int = 16
    encoding_name: str = "cl100k_base"

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("HF_TOKEN", None)


@dataclass
class InferenceCall:
    """
    A class that represents an inference call to a model.

    Attributes:
        messages: List of message dictionaries in the format expected by the LLM API.
        temperature: Optional sampling temperature for controlling randomness in generation.
        tags: List of string tags that can be set to any values by the user. Used internally
              for logging and cost tracking purposes (e.g., pipeline stage).
        max_retries: Maximum number of retry attempts for failed inference calls.
        seed: Optional random seed for reproducible outputs.
    """

    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    tags: List[str] = field(default_factory=lambda: ["dev"])  # Tags will identify the 'stage'
    max_retries: int = 8
    seed: Optional[int] = None


@dataclass
class InferenceJob:
    inference_calls: List[InferenceCall]


def _ensure_logs_dir():
    """Ensures the logs directory exists."""
    os.makedirs("logs", exist_ok=True)


def _get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Gets a tiktoken encoding, defaulting to cl100k_base with fallback."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to get encoding '{encoding_name}'. Falling back to 'cl100k_base'. Error: {e}")
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    """Counts tokens in a single string."""
    if not text:
        return 0
    try:
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return 0


def _count_message_tokens(messages: List[Dict[str, str]], encoding: tiktoken.Encoding) -> int:
    """Counts tokens in a list of messages, approximating OpenAI's format."""
    num_tokens = 0
    # Approximation based on OpenAI's cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    # This might not be perfectly accurate for all models/providers but is a reasonable estimate.
    tokens_per_message = 3
    tokens_per_name = 1

    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if value:
                num_tokens += _count_tokens(str(value), encoding)
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def _log_individual_call(model_name: str, input_tokens: int, output_tokens: int, tags: List[str], encoding_name: str):
    """Logs a single inference call's cost details."""
    global _individual_header_written
    try:
        _ensure_logs_dir()
        is_new_file = not os.path.exists(_individual_log_file)
        mode = "a" if not is_new_file else "w"

        with open(_individual_log_file, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header only if the file is new or header wasn't written yet in this run
            if is_new_file or not _individual_header_written:
                writer.writerow(["timestamp", "model_name", "stage", "input_tokens", "output_tokens", "encoding_used"])
                _individual_header_written = True

            stage = ";".join(tags) if tags else "unknown"
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            writer.writerow([timestamp, model_name, stage, input_tokens, output_tokens, encoding_name])
    except Exception as e:
        logger.error(f"Failed to write to individual cost log: {e}")


def _update_aggregate_cost(model_name: str, input_tokens: int, output_tokens: int):
    """Updates the global dictionary for aggregate costs."""
    try:
        _cost_data[model_name]["input_tokens"] += input_tokens
        _cost_data[model_name]["output_tokens"] += output_tokens
        _cost_data[model_name]["calls"] += 1
    except Exception as e:
        logger.error(f"Failed to update aggregate cost data: {e}")


def _write_aggregate_log():
    """Writes the aggregated cost data to a file at program exit."""
    try:
        if not _cost_data:
            logger.info("No cost data collected, skipping aggregate log.")
            return

        _ensure_logs_dir()
        logger.info(f"Writing aggregate cost log to {_aggregate_log_file}")
        with open(_aggregate_log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model_name", "total_input_tokens", "total_output_tokens", "total_calls"])
            for model_name, data in sorted(_cost_data.items()):
                writer.writerow([model_name, data["input_tokens"], data["output_tokens"], data["calls"]])
        logger.success(f"Aggregate cost log successfully written to {_aggregate_log_file}")
    except Exception as e:
        # Use print here as logger might be shutting down during atexit
        print(f"ERROR: Failed to write aggregate cost log: {e}", flush=True)


# Register the aggregate log function to run at exit
atexit.register(_write_aggregate_log)


async def _get_response(model: Model, inference_call: InferenceCall) -> str:
    """
    Send one inference call to the model endpoint within a global timeout context.
    Logs start/end times for better concurrency tracing and tracks token costs.
    """
    start_time = time.time()
    logger.debug(
        "START _get_response: model='{}' (encoding='{}') (timestamp={:.4f})",
        model.model_name,
        model.encoding_name,
        start_time,
    )

    request_id = str(uuid.uuid4())

    client = AsyncInferenceClient(
        base_url=model.base_url,
        api_key=model.api_key,
        provider=model.provider,
        bill_to=model.bill_to,
        timeout=GLOBAL_TIMEOUT,
        headers={"X-Request-ID": request_id},
    )

    logger.debug(f"Making request with ID: {request_id}")

    response = await client.chat_completion(
        model=model.model_name,
        messages=inference_call.messages,
        temperature=inference_call.temperature,
        # Note: seed is not directly supported by chat_completion in huggingface_hub client API as of recent versions
        # It might need to be passed via extra_body if the provider supports it.
        # seed=inference_call.seed, # This might cause an error if not supported
    )

    # Safe-guarding in case the response is missing .choices
    if not response or not response.choices:
        logger.error("Empty response or missing .choices from model {}", model.model_name)
        raise Exception("Failed Inference")

    output_content = response.choices[0].message.content

    try:
        encoding = _get_encoding(model.encoding_name)
        input_tokens = _count_message_tokens(inference_call.messages, encoding)
        output_tokens = _count_tokens(output_content, encoding)

        _log_individual_call(model.model_name, input_tokens, output_tokens, inference_call.tags, model.encoding_name)
        _update_aggregate_cost(model.model_name, input_tokens, output_tokens)
        logger.debug(f"Cost tracked: Model={model.model_name}, Input={input_tokens}, Output={output_tokens}")
    except Exception as cost_e:
        logger.error(f"Error during cost tracking for model {model.model_name}: {cost_e}")

    finish_time = time.time()
    logger.debug(
        "END _get_response: model='{}' (timestamp={:.4f}, duration={:.2f}s)",
        model.model_name,
        finish_time,
        (finish_time - start_time),
    )
    logger.debug(
        "Response content from model {} = {}",
        model.model_name,
        output_content,
    )
    return output_content


async def _retry_with_backoff(model: Model, inference_call: InferenceCall, semaphore: asyncio.Semaphore) -> str:
    """
    Attempt to get the model's response with a simple exponential backoff,
    while respecting the concurrency limit via 'semaphore'.
    Logs concurrency acquisitions and re-tries.
    """
    for attempt in range(inference_call.max_retries):
        # We log the attempt count
        logger.debug(
            "Attempt {} of {} for model '{}', waiting for semaphore...",
            attempt + 1,
            inference_call.max_retries,
            model.model_name,
        )
        async with semaphore:  # enforce concurrency limit per-model
            try:
                logger.debug(
                    "Semaphore acquired for model='{}' on attempt={} (max_concurrent={}).",
                    model.model_name,
                    attempt + 1,
                    model.max_concurrent_requests,
                )
                return await _get_response(model, inference_call)  # Cost tracking happens inside _get_response
            except Exception as e:
                logger.error("Error invoking model {}: {}", model.model_name, e)

        # Only sleep if not on the last attempt
        if attempt < inference_call.max_retries - 1:
            backoff_secs = 2 ** (attempt + 2)  # Exponential backoff (4, 8, 16, ...)
            logger.debug("Backing off for {} seconds before next attempt...", backoff_secs)
            await asyncio.sleep(backoff_secs)

    logger.critical(
        "Failed to get response from model {} after {} attempts",
        model.model_name,
        inference_call.max_retries,
    )

    try:
        encoding = _get_encoding(model.encoding_name)
        input_tokens = _count_message_tokens(inference_call.messages, encoding)
        _log_individual_call(model.model_name, input_tokens, 0, ["FAILED"] + inference_call.tags, model.encoding_name)
        _update_aggregate_cost(model.model_name, input_tokens, 0)
        logger.warning(f"Logged failed call for {model.model_name} with input tokens {input_tokens}, output 0.")
    except Exception as cost_e:
        logger.error(f"Error during cost tracking for *failed* call {model.model_name}: {cost_e}")

    return ""


async def _run_inference_async_helper(
    models: List[Model], inference_calls: List[InferenceCall]
) -> Dict[str, List[str]]:
    """
    Launch tasks for each (model, inference_call) pair in parallel, respecting concurrency.

    Returns:
        Dict[str, List[str]]: A dictionary keyed by model.model_name, each value
        is a list of responses (strings) in the same order as 'inference_calls'.
    """
    logger.info("Starting asynchronous inference with per-model concurrency control.")

    # Instead of a single global concurrency, create a semaphore per model based on model.max_concurrent_requests
    model_semaphores: Dict[str, asyncio.Semaphore] = {}
    for model in models:
        # If not specified, default to something reasonable like 1
        concurrency = max(model.max_concurrent_requests, 1)
        semaphore = asyncio.Semaphore(concurrency)
        model_semaphores[model.model_name] = semaphore
        logger.debug(
            "Created semaphore for model='{}' with concurrency={}",
            model.model_name,
            concurrency,
        )

    tasks = []
    # We'll build tasks in an order that ensures each model gets a contiguous
    # slice in the final results.
    for model in models:
        semaphore = model_semaphores[model.model_name]
        for call in inference_calls:
            tasks.append(_retry_with_backoff(model, call, semaphore))

    logger.info(
        "Total tasks scheduled: {}  (models={}  x  calls={})",
        len(tasks),
        len(models),
        len(inference_calls),
    )

    # Run them all concurrently
    results = await tqdm_asyncio.gather(*tasks)
    logger.success("Completed parallel inference for all models.")

    # Re-map results back to {model_name: [list_of_responses]}
    responses: Dict[str, List[str]] = {}
    idx = 0
    n_calls = len(inference_calls)
    for model in models:
        slice_end = idx + n_calls
        model_responses = results[idx:slice_end]
        responses[model.model_name] = model_responses
        idx = slice_end

    # Optional debug: confirm each model's result count
    for model in models:
        logger.debug(
            "Model '{}' produced {} responses.",
            model.model_name,
            len(responses[model.model_name]),
        )

    return responses


def _load_models(base_config: Dict[str, Any], step_name: str) -> List[Model]:
    """
    Load only the models assigned to this step from the config's 'model_list' and 'model_roles'.
    If no model role is defined for the step, use the first model from model_list.
    """
    all_configured_models = base_config.get("model_list", [])
    role_models = base_config.get("model_roles", {}).get(step_name, [])

    # If no role models are defined for this step, use the first model from model_list
    if not role_models and all_configured_models:
        first_model_config = all_configured_models[0]
        logger.info(
            "No models defined in model_roles for step '{}'. Using the first model from model_list: {}",
            step_name,
            first_model_config["model_name"],
        )
        return [
            Model(**{**first_model_config, "encoding_name": first_model_config.get("encoding_name", "cl100k_base")})
        ]

    # Filter out only those with a matching 'model_name'
    matched = []
    for m_config in all_configured_models:
        if m_config["model_name"] in role_models:
            model_instance = Model(**{**m_config, "encoding_name": m_config.get("encoding_name", "cl100k_base")})
            matched.append(model_instance)

    logger.info(
        "Found {} models in config for step '{}': {}",
        len(matched),
        step_name,
        [m.model_name for m in matched],
    )
    return matched


def run_inference(
    config: Dict[str, Any], step_name: str, inference_calls: List[InferenceCall]
) -> Dict[str, List[str]]:
    """
    Run inference in parallel for the given step_name and inference_calls.

    Returns a dictionary of the form:
        {
            "model_name_1": [resp_for_call_1, resp_for_call_2, ... ],
            "model_name_2": [...],
            ...
        }
    """
    # 1. Load relevant models for the pipeline step
    models = _load_models(config, step_name)
    if not models:
        logger.warning("No models found for step '{}'. Returning empty dictionary.", step_name)
        return {}

    # Assign the step_name as a tag if not already present (for cost tracking)
    for call in inference_calls:
        if step_name not in call.tags:
            call.tags.append(step_name)

    # 2. Run the concurrency-enabled async helper
    try:
        return asyncio.run(_run_inference_async_helper(models, inference_calls))
    except Exception as e:
        logger.critical("Error running inference for step '{}': {}", step_name, e)
        # Ensure aggregate log is attempted even on critical error during run
        # Note: atexit should handle this, but adding a safeguard doesn't hurt
        # _write_aggregate_log() # Redundant due to atexit
        return {}  # Return empty on failure
