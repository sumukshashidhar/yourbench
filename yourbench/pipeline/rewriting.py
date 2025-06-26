"""
Question question_rewriting Pipeline Stage

This module implements a stage that takes generated questions (both single-hop and multi-hop)
and rewrites them using an LLM while preserving their meaning and answerability.

Features:
- Preserves question meaning and answerability
- Maintains all metadata from original questions
- Works with both single-hop and multi-hop questions
- Configurable question_rewriting instructions
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from loguru import logger

from datasets import Dataset
from yourbench.utils.prompts import QUESTION_question_rewriting_USER_PROMPT, QUESTION_REWRITING_SYSTEM_PROMPT
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.parsing_engine import extract_content_from_xml_tags
from yourbench.utils.inference.inference_core import InferenceCall, run_inference
from yourbench.utils.question_models import QuestionRow


@dataclass
class RewrittenQuestion:
    """Container for a rewritten question with metadata."""

    original_question: str
    rewritten_question: str
    question_rewriting_model: str
    question_rewriting_rationale: str


def _parse_question_rewriting_response(response: str) -> Optional[RewrittenQuestion]:
    """
    Parse the model's question_rewriting response to extract the rewritten question and rationale.

    Args:
        response: Raw model response

    Returns:
        RewrittenQuestion object or None if parsing fails
    """
    try:
        rewritten_q = extract_content_from_xml_tags(response, "rewritten_question")
        rationale = extract_content_from_xml_tags(response, "question_rewriting_rationale")

        if not rewritten_q:
            logger.warning("No rewritten question found in response")
            return None

        return RewrittenQuestion(
            original_question="",  # Will be filled by caller
            rewritten_question=rewritten_q.strip(),
            question_rewriting_model="",  # Will be filled by caller
            question_rewriting_rationale=rationale.strip() if rationale else "",
        )
    except Exception as e:
        logger.error(f"Error parsing question_rewriting response: {e}")
        return None


def _build_question_rewriting_calls(
    dataset: Dataset, system_prompt: str, additional_instructions: str
) -> tuple[List[InferenceCall], List[int]]:
    """
    Build inference calls for question_rewriting questions.

    Returns:
        Tuple of (inference_calls, row_indices)
    """
    calls = []
    indices = []

    for idx, row in enumerate(dataset):
        # Extract relevant fields
        question = row.get("question", "")
        if not question:
            logger.warning(f"Skipping row {idx} - no question found")
            continue

        # Get chunks based on question type
        chunks_data = row.get("chunks", "")
        if isinstance(chunks_data, list):
            # For both multihop and single-hop, if chunks are a list, join them.
            # This correctly handles empty, single-item, and multi-item lists.
            # We use map(str, ...) to safely handle any non-string elements.
            chunk_text = "\n\n".join(map(str, chunks_data))
        else:
            # For single-hop, chunks might be a single item (e.g., a string).
            # We convert it to a string. Falsy values (like None or empty string) will result in an empty string.
            chunk_text = str(chunks_data) if chunks_data else ""

        summary = row.get("document_summary", "")
        answer = row.get("self_answer", "")

        # Build user prompt
        user_prompt = QUESTION_question_rewriting_USER_PROMPT.format(
            original_question=question,
            answer=answer,
            chunk_text=chunk_text,
            document_summary=summary,
            additional_instructions=additional_instructions,
        )

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        calls.append(InferenceCall(messages=messages, tags=["question_rewriting"]))
        indices.append(idx)

    return calls, indices


def _process_question_rewriting_responses(
    responses: Dict[str, List[str]], indices: List[int], original_dataset: Dataset
) -> List[Dict[str, Any]]:
    """
    Process model responses and create rewritten dataset rows.
    """
    rewritten_rows = []

    for model_name, model_responses in responses.items():
        if len(model_responses) != len(indices):
            logger.warning(
                f"Response count mismatch for model {model_name}. "
                f"Expected {len(indices)} but got {len(model_responses)}. "
                "This can happen if some inference calls failed. "
                "Processing the responses that were returned."
            )

        for response, dataset_idx in zip(model_responses, indices):
            if not response:
                logger.warning(f"Skipping failed or empty response for original dataset row {dataset_idx}")
                continue

            original_row = original_dataset[dataset_idx]

            # Parse the question_rewriting response
            rewritten = _parse_question_rewriting_response(response)
            if not rewritten:
                logger.warning(f"Failed to parse response for row {dataset_idx} - skipping this row")
                continue

            # Create new row with all original data plus question_rewriting info
            new_row_dict = dict(original_row)
            new_row_dict.update(
                {
                    "original_question": original_row["question"],
                    "question": rewritten.rewritten_question,
                    "question_rewriting_model": model_name,
                    "question_rewriting_rationale": rewritten.question_rewriting_rationale,
                    "raw_question_rewriting_response": response,
                }
            )

            try:
                # Validate and structure the data using QuestionRow
                question_row = QuestionRow(**new_row_dict)
                rewritten_rows.append(question_row.to_dict())
            except (TypeError, ValueError) as e:
                logger.warning(f"Skipping row {dataset_idx} due to validation error: {e}")
                logger.debug(f"Row data: {new_row_dict}")

    return rewritten_rows


def run(config: Dict[str, Any]) -> None:
    """
    Main entry point for the question_rewriting pipeline stage.

    This stage:
    1. Loads single-hop and multi-hop question datasets
    2. Sends each question to an LLM for question_rewriting
    3. Parses the rewritten questions
    4. Saves new datasets with rewritten questions
    """
    stage_cfg = config.get("pipeline", {}).get("question_rewriting", {})
    if not stage_cfg.get("run", False):
        logger.info("question_rewriting stage is disabled. Skipping.")
        return

    logger.info("Starting question question_rewriting stage...")

    # Get configuration
    additional_instructions = stage_cfg.get(
        "additional_instructions",
        "Rewrite the question to sound more natural and conversational while preserving the exact meaning.",
    )

    # Process single-hop questions
    try:
        logger.info("Processing single-hop questions...")
        single_hop_ds = custom_load_dataset(config=config, subset="single_shot_questions")

        if single_hop_ds and len(single_hop_ds) > 0:
            calls, indices = _build_question_rewriting_calls(
                single_hop_ds, QUESTION_REWRITING_SYSTEM_PROMPT, additional_instructions
            )

            if calls:
                responses = run_inference(config=config, step_name="question_rewriting", inference_calls=calls)

                rewritten_rows = _process_question_rewriting_responses(responses, indices, single_hop_ds)

                if rewritten_rows:
                    rewritten_ds = Dataset.from_list(rewritten_rows)
                    custom_save_dataset(dataset=rewritten_ds, config=config, subset="single_shot_questions_rewritten")
                    logger.success(f"Saved {len(rewritten_rows)} rewritten single-hop questions")
            else:
                logger.warning("No valid single-hop questions to rewrite")
        else:
            logger.warning("No single-hop questions found")

    except Exception as e:
        logger.error(f"Error processing single-hop questions: {e}")

    # Process multi-hop questions
    try:
        logger.info("Processing multi-hop questions...")
        multi_hop_ds = custom_load_dataset(config=config, subset="multi_hop_questions")

        if multi_hop_ds and len(multi_hop_ds) > 0:
            calls, indices = _build_question_rewriting_calls(
                multi_hop_ds, QUESTION_REWRITING_SYSTEM_PROMPT, additional_instructions
            )

            if calls:
                responses = run_inference(config=config, step_name="question_rewriting", inference_calls=calls)

                rewritten_rows = _process_question_rewriting_responses(responses, indices, multi_hop_ds)

                if rewritten_rows:
                    rewritten_ds = Dataset.from_list(rewritten_rows)
                    custom_save_dataset(dataset=rewritten_ds, config=config, subset="multi_hop_questions_rewritten")
                    logger.success(f"Saved {len(rewritten_rows)} rewritten multi-hop questions")
            else:
                logger.warning("No valid multi-hop questions to rewrite")
        else:
            logger.warning("No multi-hop questions found")

    except Exception as e:
        logger.error(f"Error processing multi-hop questions: {e}")

    logger.success("Question question_rewriting stage completed")
