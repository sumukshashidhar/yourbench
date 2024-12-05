from dotenv import load_dotenv
from typing import List
import litellm
from tqdm import tqdm

load_dotenv()

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]


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
