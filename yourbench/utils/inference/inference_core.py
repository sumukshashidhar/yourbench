import os
import time
import uuid
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import field, dataclass

from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from huggingface_hub import AsyncInferenceClient
from yourbench.utils.inference.inference_tracking import (
    InferenceMetrics,
    _count_tokens,
    _get_encoding,
    _categorize_error,
    _count_message_tokens,
    log_inference_metrics,
    get_performance_summary,
    update_aggregate_metrics,
)


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
    encoding_name: str = "cl100k_base"
    extra_parameters: Dict[str, Any] = field(default_factory=dict)

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
    max_retries: int = 12
    seed: Optional[int] = None
    extra_parameters: Dict[str, Any] = field(default_factory=dict)


def _load_models(base_config, step_name: str) -> List[Model]:
    """
    Load only the models assigned to this step from the config's 'model_list' and 'model_roles'.
    If no model role is defined for the step, use the first model from model_list.
    """
    all_configured_models = base_config.model_list
    role_models = base_config.model_roles.get(step_name, [])

    # If no role models are defined for this step, use the first model from model_list
    if not role_models and all_configured_models:
        first_model_config = all_configured_models[0]
        logger.info(
            "No models defined in model_roles for step '{}'. Using the first model from model_list: {}",
            step_name,
            first_model_config.model_name,
        )
        return [
            Model(
                model_name=first_model_config.model_name,
                provider=first_model_config.provider,
                base_url=first_model_config.base_url,
                api_key=first_model_config.api_key,
                bill_to=first_model_config.bill_to,
                max_concurrent_requests=first_model_config.max_concurrent_requests,
                encoding_name=first_model_config.encoding_name,
                extra_parameters=dict(first_model_config.extra_parameters or {}),
            )
        ]

    # Filter out only those with a matching 'model_name'
    matched = []
    for m_config in all_configured_models:
        if m_config.model_name in role_models:
            model_instance = Model(
                model_name=m_config.model_name,
                provider=m_config.provider,
                base_url=m_config.base_url,
                api_key=m_config.api_key,
                bill_to=m_config.bill_to,
                max_concurrent_requests=m_config.max_concurrent_requests,
                encoding_name=m_config.encoding_name,
                extra_parameters=dict(m_config.extra_parameters or {}),
            )
            matched.append(model_instance)

    logger.info(
        "Found {} models in config for step '{}': {}",
        len(matched),
        step_name,
        [m.model_name for m in matched],
    )
    return matched


async def _get_response(
    model: Model,
    inference_call: InferenceCall,
    request_id: str = None,
    concurrency_level: int = 1,
    queue_start_time: float = None,
) -> tuple[str, InferenceMetrics]:
    """
    Send one inference call to the model endpoint with comprehensive metrics tracking.
    """
    start_time = time.time()
    request_id = request_id or str(uuid.uuid4())
    queue_time = (start_time - queue_start_time) if queue_start_time else 0.0

    logger.debug(
        "START _get_response: model='{}' request_id='{}' (encoding='{}') (timestamp={:.4f})",
        model.model_name,
        request_id,
        model.encoding_name,
        start_time,
    )

    # Initialize metrics
    encoding = _get_encoding(model.encoding_name)
    input_tokens = _count_message_tokens(inference_call.messages, encoding)
    stage = ";".join(inference_call.tags) if inference_call.tags else "unknown"

    metrics = InferenceMetrics(
        request_id=request_id,
        model_name=model.model_name,
        stage=stage,
        input_tokens=input_tokens,
        output_tokens=0,
        duration=0.0,
        queue_time=queue_time,
        retry_count=0,
        success=False,
        concurrency_level=concurrency_level,
        temperature=inference_call.temperature,
        encoding_name=model.encoding_name,
    )

    try:
        client = AsyncInferenceClient(
            base_url=model.base_url,
            api_key=model.api_key,
            provider=model.provider,
            bill_to=model.bill_to,
            timeout=GLOBAL_TIMEOUT,
            headers={"X-Request-ID": request_id},
        )

        logger.debug(f"Making request with ID: {request_id}")
        extra_body: Dict[str, Any] | None = None
        if model.extra_parameters:
            extra_body = dict(model.extra_parameters)
        if inference_call.extra_parameters:
            if extra_body is None:
                extra_body = {}
            extra_body.update(inference_call.extra_parameters)

        chat_kwargs: Dict[str, Any] = {
            "model": model.model_name,
            "messages": inference_call.messages,
        }
        if inference_call.temperature is not None:
            chat_kwargs["temperature"] = inference_call.temperature
        if extra_body:
            chat_kwargs["extra_body"] = extra_body

        response = await client.chat_completion(**chat_kwargs)

        # Safe-guarding in case the response is missing .choices
        if not response or not response.choices:
            error_msg = f"Empty response or missing .choices from model {model.model_name}"
            logger.warning(error_msg)
            raise Exception(error_msg)

        output_content = response.choices[0].message.content
        finish_time = time.time()

        # Update metrics for successful call
        metrics.output_tokens = _count_tokens(output_content, encoding)
        metrics.duration = finish_time - start_time
        metrics.success = True

        logger.debug(
            "END _get_response: model='{}' request_id='{}' (timestamp={:.4f}, duration={:.2f}s, tokens={}/{})",
            model.model_name,
            request_id,
            finish_time,
            metrics.duration,
            metrics.input_tokens,
            metrics.output_tokens,
        )

        return output_content, metrics

    except Exception as e:
        finish_time = time.time()
        metrics.duration = finish_time - start_time
        metrics.success = False
        metrics.error_type = _categorize_error(e)
        metrics.error_message = str(e)[:500]  # Truncate long error messages

        logger.warning(
            "ERROR _get_response: model='{}' request_id='{}' error_type='{}' duration={:.2f}s: {}",
            model.model_name,
            request_id,
            metrics.error_type,
            metrics.duration,
            str(e)[:100],
        )

        raise e
    finally:
        # Always log metrics, whether successful or not
        log_inference_metrics(metrics)
        update_aggregate_metrics(
            model.model_name,
            metrics.input_tokens,
            metrics.output_tokens,
            metrics.duration,
            metrics.success,
            metrics.queue_time,
            metrics.retry_count,
            error=Exception(metrics.error_message) if metrics.error_message else None,
            concurrency_level=concurrency_level,
        )


async def _retry_with_backoff(
    model: Model, inference_call: InferenceCall, semaphore: asyncio.Semaphore, concurrency_level: int
) -> str:
    """
    Attempt to get the model's response with exponential backoff and comprehensive tracking.
    """
    queue_start_time = time.time()
    request_id = str(uuid.uuid4())

    for attempt in range(inference_call.max_retries):
        logger.debug(
            "Attempt {} of {} for model '{}' request_id='{}', waiting for semaphore...",
            attempt + 1,
            inference_call.max_retries,
            model.model_name,
            request_id,
        )

        semaphore_wait_start = time.time()
        async with semaphore:
            semaphore_wait_time = time.time() - semaphore_wait_start

            logger.debug(
                "Semaphore acquired for model='{}' request_id='{}' on attempt={} (wait_time={:.2f}s, max_concurrent={}).",
                model.model_name,
                request_id,
                attempt + 1,
                semaphore_wait_time,
                model.max_concurrent_requests,
            )

            try:
                # Calculate actual queue time including semaphore wait
                actual_queue_start = queue_start_time if attempt == 0 else semaphore_wait_start
                output_content, metrics = await _get_response(
                    model, inference_call, request_id, concurrency_level, actual_queue_start
                )

                # Update retry count in metrics
                metrics.retry_count = attempt

                # Log success summary
                logger.debug(
                    "SUCCESS: model='{}' request_id='{}' after {} attempts (total_time={:.2f}s, tokens={}/{})",
                    model.model_name,
                    request_id,
                    attempt + 1,
                    time.time() - queue_start_time,
                    metrics.input_tokens,
                    metrics.output_tokens,
                )

                return output_content

            except Exception as e:
                attempt_error = _categorize_error(e)
                logger.warning(
                    "Attempt {} failed for model '{}' request_id='{}': {} ({})",
                    attempt + 1,
                    model.model_name,
                    request_id,
                    attempt_error,
                    str(e)[:100],
                )

        # Only sleep if not on the last attempt
        if attempt < inference_call.max_retries - 1:
            backoff_secs = 2 ** (attempt + 2)  # Exponential backoff (4, 8, 16, ...)
            logger.debug(
                "Backing off for {} seconds before next attempt for request_id='{}'...",
                backoff_secs,
                request_id,
            )
            await asyncio.sleep(backoff_secs)

    # All attempts failed
    total_time = time.time() - queue_start_time
    logger.critical(
        "FAILED: model='{}' request_id='{}' after {} attempts (total_time={:.2f}s)",
        model.model_name,
        request_id,
        inference_call.max_retries,
        total_time,
    )

    # Log final failure metrics
    try:
        encoding = _get_encoding(model.encoding_name)
        input_tokens = _count_message_tokens(inference_call.messages, encoding)
        stage = ";".join(inference_call.tags) if inference_call.tags else "unknown"

        failed_metrics = InferenceMetrics(
            request_id=request_id,
            model_name=model.model_name,
            stage=stage,
            input_tokens=input_tokens,
            output_tokens=0,
            duration=total_time,
            queue_time=0.0,
            retry_count=inference_call.max_retries,
            success=False,
            error_type="max_retries_exceeded",
            error_message=f"Failed after {inference_call.max_retries} attempts",
            concurrency_level=concurrency_level,
            temperature=inference_call.temperature,
            encoding_name=model.encoding_name,
        )

        log_inference_metrics(failed_metrics)
        update_aggregate_metrics(
            model.model_name,
            input_tokens,
            0,
            total_time,
            False,
            0.0,
            inference_call.max_retries,
            Exception("Max retries exceeded"),
            concurrency_level,
        )

    except Exception as metrics_error:
        logger.error(f"Error logging failure metrics for {model.model_name}: {metrics_error}")

    return ""


async def _run_inference_async_helper(
    models: List[Model], inference_calls: List[InferenceCall]
) -> Dict[str, List[str]]:
    """
    Launch tasks for each (model, inference_call) pair in parallel with enhanced tracking.
    """
    logger.info("Starting asynchronous inference with enhanced tracking and per-model concurrency control.")

    # Create semaphores with tracking
    model_semaphores: Dict[str, asyncio.Semaphore] = {}
    for model in models:
        concurrency = max(model.max_concurrent_requests, 1)
        semaphore = asyncio.Semaphore(concurrency)
        model_semaphores[model.model_name] = semaphore
        logger.debug(
            "Created semaphore for model='{}' with concurrency={}",
            model.model_name,
            concurrency,
        )

    tasks = []
    total_start_time = time.time()

    # Build tasks with concurrency level tracking
    for model in models:
        semaphore = model_semaphores[model.model_name]
        concurrency_level = model.max_concurrent_requests

        for call in inference_calls:
            task = _retry_with_backoff(model, call, semaphore, concurrency_level)
            tasks.append(task)

    logger.info(
        "Total tasks scheduled: {} (models={} x calls={})",
        len(tasks),
        len(models),
        len(inference_calls),
    )

    # Run all tasks concurrently with progress tracking
    results = await tqdm_asyncio.gather(*tasks, desc="Running inference")

    total_duration = time.time() - total_start_time
    logger.success(
        "Completed parallel inference for all models in {:.2f}s (avg {:.2f}s per task)",
        total_duration,
        total_duration / len(tasks) if tasks else 0,
    )

    # Log performance summaries
    for model in models:
        summary = get_performance_summary(model.model_name)
        if summary:
            logger.info(
                "Performance summary for {}: success_rate={:.2%}, avg_duration={:.2f}s, "
                "avg_tokens_in/out={:.0f}/{:.0f}, retry_rate={:.2f}",
                model.model_name,
                summary["success_rate"],
                summary["avg_duration"],
                summary["avg_request_size"],
                summary["avg_response_size"],
                summary["avg_retry_count"],
            )

    # Re-map results back to {model_name: [list_of_responses]}
    responses: Dict[str, List[str]] = {}
    idx = 0
    n_calls = len(inference_calls)
    for model in models:
        slice_end = idx + n_calls
        model_responses = results[idx:slice_end]
        responses[model.model_name] = model_responses
        idx = slice_end

    # Log final response counts
    for model in models:
        successful_responses = len([r for r in responses[model.model_name] if r])
        logger.debug(
            "Model '{}' produced {}/{} successful responses.",
            model.model_name,
            successful_responses,
            len(responses[model.model_name]),
        )

    return responses


def run_inference(
    config, step_name: str, inference_calls: List[InferenceCall]
) -> Dict[str, List[str]]:
    """
    Run inference in parallel for the given step_name and inference_calls with enhanced tracking.

    Returns a dictionary of the form:
        {
            "model_name_1": [resp_for_call_1, resp_for_call_2, ... ],
            "model_name_2": [...],
            ...
        }
    """
    logger.info(f"Starting inference for step '{step_name}' with {len(inference_calls)} calls")

    # Load relevant models for the pipeline step
    models = _load_models(config, step_name)
    if not models:
        logger.warning("No models found for step '{}'. Returning empty dictionary.", step_name)
        return {}

    # Assign the step_name as a tag if not already present (for tracking)
    for call in inference_calls:
        if step_name not in call.tags:
            call.tags.append(step_name)

    # Run the enhanced async helper
    try:
        start_time = time.time()
        result = asyncio.run(_run_inference_async_helper(models, inference_calls))
        total_time = time.time() - start_time

        logger.success(
            "Inference completed for step '{}' in {:.2f}s with {} models",
            step_name,
            total_time,
            len(models),
        )

        return result

    except Exception as e:
        logger.critical("Error running inference for step '{}': {}", step_name, e)
        return {}
