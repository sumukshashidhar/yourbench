import asyncio
import os

# from asyncio import timeout
from typing import List

import aiohttp
import litellm
from async_timeout import timeout
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio


load_dotenv()

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]
litellm.set_verbose = True


def get_batch_completion(
    model_name: str,
    model_type: str,
    user_prompts: List[str],
    batch_size: int = 4,
    **kwargs,
):
    """Get batch completions for a list of user prompts"""
    completions = []
    total_batches = (len(user_prompts) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(user_prompts), batch_size),
        total=total_batches,
        desc="Processing batches",
    ):
        batch_prompts = user_prompts[i : i + batch_size]
        batch_completions = litellm.batch_completion(
            model=f"{model_type}/{model_name}",
            messages=batch_prompts,
            **kwargs,
        )
        completions.extend(batch_completions)
    final_responses = []
    for response in completions:
        try:
            final_responses.append(response.choices[0].message.content)
        except Exception as e:
            print(f"Error processing response: {e}")
            final_responses.append("")
    return final_responses


async def _process_single_prompt(
    session: aiohttp.ClientSession,
    prompt: dict,
    model_type: str,
    model_name: str,
    base_url: str = None,
    api_key: str = None,
    semaphore: asyncio.Semaphore = None,
    retry_attempts: int = 6,
    timeout_seconds: int = 600,
) -> dict:
    async with semaphore:
        for attempt in range(retry_attempts):
            try:
                async with timeout(timeout_seconds):  # 30 second timeout
                    response = await litellm.acompletion(
                        model=f"{model_type}/{model_name}",
                        messages=prompt,
                        api_base=base_url,
                        api_key=api_key,
                        max_tokens=4096,
                    )
                    return response.choices[0].message.content
            except Exception as e:
                if attempt == retry_attempts - 1:
                    print(f"Failed after {retry_attempts} attempts: {e}")
                    return ""
                await asyncio.sleep(2**attempt)  # Exponential backoff


async def perform_parallel_inference(prompts: List[dict], config: dict):
    """Perform parallel inference on a list of prompts with order preservation"""
    selected_model = config["configurations"]["model"]
    model_name = selected_model["model_name"]
    model_type = selected_model["model_type"]
    model_base_url = (
        os.getenv("MODEL_BASE_URL")
        if model_type == "openai"
        else os.getenv("AZURE_BASE_URL")
    )
    model_api_key = (
        os.getenv("MODEL_API_KEY")
        if model_type == "openai"
        else os.getenv("AZURE_API_KEY")
    )
    # Control concurrency with a semaphore (adjust based on API limits)
    max_concurrent = selected_model["max_concurrent_requests"]
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = [
            _process_single_prompt(
                session,
                prompt,
                model_type,
                model_name,
                model_base_url,
                model_api_key,
                semaphore,
            )
            for prompt in prompts
        ]

        # Use tqdm_asyncio to show progress while maintaining order
        results = await tqdm_asyncio.gather(
            *tasks, desc="Processing prompts", total=len(prompts)
        )

    return results


# Add this to use the async function
def run_parallel_inference(prompts: List[dict], config: dict):
    """Synchronous wrapper for the async function"""
    return asyncio.run(perform_parallel_inference(prompts, config))
