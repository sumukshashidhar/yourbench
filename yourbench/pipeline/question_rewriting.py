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
from yourbench.utils.prompts import QUESTION_REWRITING_SYSTEM_PROMPT, QUESTION_question_rewriting_USER_PROMPT
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.parsing_engine import extract_content_from_xml_tags
from yourbench.utils.question_models import QuestionRow
from yourbench.utils.inference.inference_core import InferenceCall, run_inference

STAGE_TAG = ["question_rewriting"]

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

        calls.append(InferenceCall(messages=messages, tags=[STAGE_TAG]))
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
            new_row_dict.update({
                "original_question": original_row["question"],
                "question": rewritten.rewritten_question,
                "question_rewriting_model": model_name,
                "question_rewriting_rationale": rewritten.question_rewriting_rationale,
                "raw_question_rewriting_response": response,
            })

            try:
                # Validate and structure the data using QuestionRow
                question_row = QuestionRow(**new_row_dict)
                rewritten_rows.append(question_row.to_dict())
            except (TypeError, ValueError) as e:
                logger.warning(f"Skipping row {dataset_idx} due to validation error: {e}")
                logger.debug(f"Row data: {new_row_dict}")

    return rewritten_rows


def _process_question_type(
    config: Dict[str, Any],
    question_type: str,
    load_subset: str,
    save_subset: str,
    additional_instructions: str,
) -> None:
    """
    Loads, rewrites, and saves a specific type of questions.

    Args:
        config: The main configuration dictionary.
        question_type: A string describing the question type for logging (e.g., "single-hop").
        load_subset: The dataset subset to load questions from.
        save_subset: The dataset subset to save rewritten questions to.
        additional_instructions: Instructions for the rewriting model.
    """
    try:
        logger.info(f"Processing {question_type} questions...")
        dataset = custom_load_dataset(config=config, subset=load_subset)

        if not dataset or len(dataset) == 0:
            logger.warning(f"No {question_type} questions found or dataset is empty.")
            return

        calls, indices = _build_question_rewriting_calls(
            dataset, QUESTION_REWRITING_SYSTEM_PROMPT, additional_instructions
        )

        if not calls:
            logger.warning(f"No valid {question_type} questions to rewrite.")
            return

        responses = run_inference(config=config, step_name="question_rewriting", inference_calls=calls)
        rewritten_rows = _process_question_rewriting_responses(responses, indices, dataset)

        if not rewritten_rows:
            logger.warning(f"No {question_type} questions were successfully rewritten.")
            return

        rewritten_ds = Dataset.from_list(rewritten_rows)
        custom_save_dataset(dataset=rewritten_ds, config=config, subset=save_subset)
        logger.success(f"Saved {len(rewritten_rows)} rewritten {question_type} questions.")

    except Exception as e:
        logger.error(f"Error processing {question_type} questions: {e}")


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

    additional_instructions = stage_cfg.get(
        "additional_instructions",
        "Rewrite the question to sound more natural and conversational while preserving the exact meaning.",
    )

    question_types_to_process = {
        "single-hop": ("single_shot_questions", "single_shot_questions_rewritten"),
        "multi-hop": ("multi_hop_questions", "multi_hop_questions_rewritten"),
    }

    for question_type, (load_subset, save_subset) in question_types_to_process.items():
        _process_question_type(
            config=config,
            question_type=question_type,
            load_subset=load_subset,
            save_subset=save_subset,
            additional_instructions=additional_instructions,
        )

    logger.success("Question question_rewriting stage completed")
