"""
Inference Engine For Yourbench
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
    messages: list[dict]
    tags: list[str] = field(default_factory=lambda: ['dev'])
    max_retries: int = 3
    
@dataclass
class InferenceJob:
    inference_calls: list[InferenceCall]
    
    
def _load_models(base_config: Dict[str, Any], step_name: str) -> list[Model]:
    """Load models from the config file"""
    models = []
    for model in base_config["model_list"]:
        models.append(Model(**model))
    logger.info("Loaded {} models from the config file", len(models))
    logger.info("Model roles: {}", base_config["model_roles"])
    logger.info("Step name: {}", step_name)
    logger.info("Model roles for step {}: {}", step_name, base_config["model_roles"][step_name])
    # only keep the model roles for the given step
    models = [model for model in models if model.model_name in base_config["model_roles"][step_name]]
    logger.info("Retained models: {}", models)
    return models


@asynccontextmanager
async def _timeout_context(timeout_seconds: int):
    try:
        async with aio_timeout(timeout_seconds):
            yield
    except asyncio.TimeoutError:
        logger.error("Operation timed out after {} seconds", timeout_seconds)
        raise

async def _get_response(model: Model, inference_call: InferenceCall) -> str:
    logger.debug("Invoking model: {}", model.model_name)
    async with _timeout_context(GLOBAL_TIMEOUT):
        response = await litellm.acompletion(
            model = model.provider + "/" + model.model_name,
            base_url = model.base_url,
            api_key = model.api_key,
            messages = inference_call.messages,
            metadata = {"tags": inference_call.tags}
        )
    logger.debug("Model {} returned response: {}", model.model_name, response.choices[0].message.content)
    return response.choices[0].message.content

async def _retry_with_backoff(model: Model, inference_call: InferenceCall) -> str:
    """
    Attempt to get the model's response using the provided router within a specified timeout.
    Internal helper function.
    """
    logger.debug("Invoking model {} with backoff", model.model_name)
    for i in range(inference_call.max_retries):
        try:
            logger.debug("Attempt {} of {} for model {}", i, inference_call.max_retries, model.model_name)
            return await _get_response(model, inference_call)
        except Exception as e:
            logger.error("Error invoking model {}: {}", model.model_name, e)
            if i < inference_call.max_retries - 1:  # Don't sleep on the last attempt
                await asyncio.sleep(2 ** (i + 2))
    logger.critical("Failed to get response from model {} after {} attempts", model.model_name, inference_call.max_retries)
    # fail with a blank string
    return ""

async def _run_inference_async_helper(models: list[Model], inference_calls: list[InferenceCall]) -> list[str]:
    """
    Run the inference calls asynchronously.
    """
    logger.info("Starting asynchronous inference...")
    semaphore = asyncio.Semaphore(32)
    logger.info("Semaphore set to {}".format(semaphore._value))

    logger.debug("Inference calls: {}", inference_calls)
    
    tasks = []
    for model in models:
        for inference_call in inference_calls:
            logger.debug("Adding task for model {} and inference call {}", model.model_name, inference_call)
            tasks.append(
                _retry_with_backoff(
                    model,
                    inference_call
                )
            )
    
    responses: Dict[str, List[str]] = {}
    results = await tqdm_asyncio.gather(*tasks)
    idx = 0
    for model in models:
        slice_end = idx + len(inference_calls)
        responses[model.model_name] = results[idx:slice_end]
        idx = slice_end

    logger.success("Completed parallel inference for all models.")
    return responses


def run_inference(config: Dict[str, Any], step_name: str, inference_calls: list[InferenceCall]) -> list[str]:
    """Run inference on the given config and inference calls."""
    models = _load_models(config, step_name)
    try:
        return asyncio.run(_run_inference_async_helper(models, inference_calls))
    except Exception as e:
        logger.critical("Error running inference: {}", e)
        return []