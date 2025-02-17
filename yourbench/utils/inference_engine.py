"""
Inference Engine For Yourbench - Now with true concurrency throttling.
"""

import litellm
from dataclasses import dataclass, field
from dotenv import load_dotenv
from loguru import logger
from typing import Dict, Any, List
import sys
import asyncio
from contextlib import asynccontextmanager

if sys.version_info >= (3, 11):
    from asyncio import timeout as aio_timeout
else:
    from async_timeout import timeout as aio_timeout

from tqdm.asyncio import tqdm_asyncio

load_dotenv()

GLOBAL_TIMEOUT = 600

# Optional: Customize success/failure callbacks
litellm.success_callback = ['langfuse']
litellm.failure_callback = ['langfuse']

@dataclass
class Model:
    model_name: str
    provider: str
    base_url: str
    api_key: str
    max_concurrent_requests: int

@dataclass
class InferenceCall:
    messages: List[Dict[str, str]]
    tags: List[str] = field(default_factory=lambda: ['dev'])
    max_retries: int = 3

@dataclass
class InferenceJob:
    inference_calls: List[InferenceCall]

@asynccontextmanager
async def _timeout_context(timeout_seconds: int):
    """
    Async context manager to raise a TimeoutError if code inside takes
    longer than 'timeout_seconds'.
    """
    try:
        async with aio_timeout(timeout_seconds):
            yield
    except asyncio.TimeoutError:
        logger.error("Operation timed out after {} seconds", timeout_seconds)
        raise

async def _get_response(model: Model, inference_call: InferenceCall) -> str:
    """
    Send one inference call to the model endpoint within a global timeout context.
    """
    logger.debug("Invoking model: {}", model.model_name)
    async with _timeout_context(GLOBAL_TIMEOUT):
        response = await litellm.acompletion(
            model=f"{model.provider}/{model.model_name}",
            base_url=model.base_url,
            api_key=model.api_key,
            messages=inference_call.messages,
            metadata={"tags": inference_call.tags}
        )
    # Safe-guarding in case the response is missing .choices
    if not response or not response.choices:
        logger.warning("Empty response or missing .choices from model {}", model.model_name)
        return ""
    logger.debug("Model {} returned response: {}", model.model_name, response.choices[0].message.content)
    return response.choices[0].message.content

async def _retry_with_backoff(model: Model, inference_call: InferenceCall, semaphore: asyncio.Semaphore) -> str:
    """
    Attempt to get the model's response with a simple exponential backoff,
    while respecting the concurrency limit via 'semaphore'.
    """
    for attempt in range(inference_call.max_retries):
        async with semaphore:  # enforce concurrency limit
            try:
                logger.debug("Attempt {} of {} for model {}", attempt+1, inference_call.max_retries, model.model_name)
                return await _get_response(model, inference_call)
            except Exception as e:
                logger.error("Error invoking model {}: {}", model.model_name, e)
        # Only sleep if not on the last attempt
        if attempt < inference_call.max_retries - 1:
            await asyncio.sleep(2 ** (attempt + 2))
    logger.critical("Failed to get response from model {} after {} attempts", model.model_name, inference_call.max_retries)
    return ""

async def _run_inference_async_helper(
    models: List[Model],
    inference_calls: List[InferenceCall]
) -> Dict[str, List[str]]:
    """
    Launch tasks for each (model, inference_call) pair in parallel, respecting concurrency.
    
    Returns:
        Dict[str, List[str]]: A dictionary keyed by model.model_name, each value
        is a list of responses (strings) in the same order as 'inference_calls'.
    """
    logger.info("Starting asynchronous inference with concurrency control.")

    # We can enforce a global concurrency or a per-model concurrency. Here we keep
    # it simple with one global semaphore, e.g., concurrency of 32 at once:
    global_concurrency = 32
    semaphore = asyncio.Semaphore(global_concurrency)
    logger.info("Global concurrency set to {} tasks at a time.", global_concurrency)

    tasks = []
    # We'll build tasks in an order that ensures each model gets a contiguous
    # slice in the final results.
    for model in models:
        for call in inference_calls:
            tasks.append(_retry_with_backoff(model, call, semaphore))

    # Run them all concurrently
    results = await tqdm_asyncio.gather(*tasks)
    logger.success("Completed parallel inference for all models.")

    # Re-map results back to {model_name: [list_of_responses]}
    responses: Dict[str, List[str]] = {}
    idx = 0
    n_calls = len(inference_calls)
    for model in models:
        # Each model is responsible for 'n_calls' tasks
        slice_end = idx + n_calls
        model_responses = results[idx:slice_end]
        responses[model.model_name] = model_responses
        idx = slice_end

    return responses

def _load_models(base_config: Dict[str, Any], step_name: str) -> List[Model]:
    """
    Load only the models assigned to this step from the config's 'model_list' and 'model_roles'.
    """
    all_configured_models = base_config.get("model_list", [])
    role_models = base_config["model_roles"].get(step_name, [])
    # Filter out only those with a matching 'model_name'
    matched = [
        Model(**m) for m in all_configured_models
        if m["model_name"] in role_models
    ]
    logger.info(
        "Found {} models in config, step '{}': {}",
        len(matched),
        step_name,
        [m.model_name for m in matched]
    )
    return matched

def run_inference(
    config: Dict[str, Any],
    step_name: str,
    inference_calls: List[InferenceCall]
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
