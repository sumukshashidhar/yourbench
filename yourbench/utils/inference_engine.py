"""
Inference Engine For Yourbench - Now with true concurrency throttling.
"""

import os
import time
import uuid
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import field, dataclass

from dotenv import load_dotenv
from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from huggingface_hub import AsyncInferenceClient, whoami


load_dotenv()

GLOBAL_TIMEOUT = 300


@dataclass
class Model:
    model_name: str
    # You can find the list of available providers here: https://huggingface.co/docs/huggingface_hub/guides/inference#supported-providers-and-tasks
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    bill_to: str | None = None

    max_concurrent_requests: int = 16

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("HF_TOKEN", None)
        if self.bill_to is None and self.base_url is None:
            # assume we are using HF inference providers
            self.api_key = os.getenv("HF_ORGANIZATION", whoami(token=self.api_key))


@dataclass
class InferenceCall:
    """
    A class that represents an inference call to a model.

    Attributes:
        messages: List of message dictionaries in the format expected by the LLM API.
        temperature: Optional sampling temperature for controlling randomness in generation.
        tags: List of string tags that can be set to any values by the user. Used internally
              for logging and cost tracking purposes.
        max_retries: Maximum number of retry attempts for failed inference calls.
        seed: Optional random seed for reproducible outputs.
    """

    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    tags: List[str] = field(default_factory=lambda: ["dev"])
    max_retries: int = 8
    seed: Optional[int] = None


@dataclass
class InferenceJob:
    inference_calls: List[InferenceCall]


async def _get_response(model: Model, inference_call: InferenceCall) -> str:
    """
    Send one inference call to the model endpoint within a global timeout context.
    Logs start/end times for better concurrency tracing.
    """
    start_time = time.time()
    logger.debug(
        "START _get_response: model='{}'  (timestamp={:.4f})",
        model.model_name,
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
    )

    # Safe-guarding in case the response is missing .choices
    if not response or not response.choices:
        logger.error("Empty response or missing .choices from model {}", model.model_name)
        raise Exception("Failed Inference")

    finish_time = time.time()
    logger.debug(
        "END _get_response: model='{}'  (timestamp={:.4f}, duration={:.2f}s)",
        model.model_name,
        finish_time,
        (finish_time - start_time),
    )
    logger.debug(
        "Response content from model {} = {}",
        model.model_name,
        response.choices[0].message.content,
    )
    return response.choices[0].message.content


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
                return await _get_response(model, inference_call)
            except Exception as e:
                logger.error("Error invoking model {}: {}", model.model_name, e)

        # Only sleep if not on the last attempt
        if attempt < inference_call.max_retries - 1:
            backoff_secs = 2 ** (attempt + 2)
            logger.debug("Backing off for {} seconds before next attempt...", backoff_secs)
            await asyncio.sleep(backoff_secs)

    logger.critical(
        "Failed to get response from model {} after {} attempts",
        model.model_name,
        inference_call.max_retries,
    )
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
        logger.info(
            "No models defined in model_roles for step '{}'. Using the first model from model_list: {}",
            step_name,
            all_configured_models[0]["model_name"],
        )
        return [Model(**all_configured_models[0])]

    # Filter out only those with a matching 'model_name'
    matched = [Model(**m) for m in all_configured_models if m["model_name"] in role_models]
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

    # 2. Run the concurrency-enabled async helper
    try:
        return asyncio.run(_run_inference_async_helper(models, inference_calls))
    except Exception as e:
        logger.critical("Error running inference for step '{}': {}", step_name, e)
        return {}
