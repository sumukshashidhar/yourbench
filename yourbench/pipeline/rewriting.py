"""
Question Rewriting Pipeline Stage

This module implements a stage that takes generated questions (both single-hop and multi-hop)
and rewrites them using an LLM while preserving their meaning and answerability.

Features:
- Preserves question meaning and answerability
- Maintains all metadata from original questions
- Works with both single-hop and multi-hop questions
- Configurable rewriting instructions
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from loguru import logger

from datasets import Dataset
from yourbench.utils.prompts import QUESTION_REWRITING_USER_PROMPT, QUESTION_REWRITING_SYSTEM_PROMPT
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.parsing_engine import extract_content_from_xml_tags
from yourbench.utils.inference.inference_core import InferenceCall, run_inference


@dataclass
class RewrittenQuestion:
    """Container for a rewritten question with metadata."""

    original_question: str
    rewritten_question: str
    rewriting_model: str
    rewriting_rationale: str


def _parse_rewriting_response(response: str) -> Optional[RewrittenQuestion]:
    """
    Parse the model's rewriting response to extract the rewritten question and rationale.

    Args:
        response: Raw model response

    Returns:
        RewrittenQuestion object or None if parsing fails
    """
    try:
        rewritten_q = extract_content_from_xml_tags(response, "rewritten_question")
        rationale = extract_content_from_xml_tags(response, "rewriting_rationale")

        if not rewritten_q:
            logger.warning("No rewritten question found in response")
            return None

        return RewrittenQuestion(
            original_question="",  # Will be filled by caller
            rewritten_question=rewritten_q.strip(),
            rewriting_model="",  # Will be filled by caller
            rewriting_rationale=rationale.strip() if rationale else "",
        )
    except Exception as e:
        logger.error(f"Error parsing rewriting response: {e}")
        return None


def _build_rewriting_calls(
    dataset: Dataset, system_prompt: str, additional_instructions: str, is_multihop: bool = False
) -> tuple[List[InferenceCall], List[int]]:
    """
    Build inference calls for rewriting questions.

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
        if is_multihop:
            chunks = row.get("chunks", [])
            chunk_text = "\n\n".join(chunks) if chunks else ""
        else:
            # For single-hop, chunks might be a single string or list
            chunks_data = row.get("chunks", [])
            if isinstance(chunks_data, list) and chunks_data:
                chunk_text = chunks_data[0] if len(chunks_data) == 1 else "\n\n".join(chunks_data)
            else:
                chunk_text = str(chunks_data) if chunks_data else ""

        summary = row.get("document_summary", "")
        answer = row.get("self_answer", "")

        # Build user prompt
        user_prompt = QUESTION_REWRITING_USER_PROMPT.format(
            original_question=question,
            answer=answer,
            chunk_text=chunk_text,
            document_summary=summary,
            additional_instructions=additional_instructions,
        )

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        calls.append(InferenceCall(messages=messages, tags=["rewriting"]))
        indices.append(idx)

    return calls, indices


def _process_rewriting_responses(
    responses: Dict[str, List[str]], indices: List[int], original_dataset: Dataset
) -> List[Dict[str, Any]]:
    """
    Process model responses and create rewritten dataset rows.
    """
    rewritten_rows = []

    for model_name, model_responses in responses.items():
        if len(model_responses) != len(indices):
            logger.error(f"Response count mismatch for {model_name}")
            continue

        for resp_idx, (response, dataset_idx) in enumerate(zip(model_responses, indices)):
            original_row = original_dataset[dataset_idx]

            # Parse the rewriting response
            rewritten = _parse_rewriting_response(response)
            if not rewritten:
                logger.warning(f"Failed to parse response for row {dataset_idx} - skipping this row")
                continue

            # Create new row with all original data plus rewriting info
            new_row = dict(original_row)
            new_row.update({
                "original_question": original_row["question"],
                "question": rewritten.rewritten_question,
                "rewriting_model": model_name,
                "rewriting_rationale": rewritten.rewriting_rationale,
                "raw_rewriting_response": response,
            })

            rewritten_rows.append(new_row)

    return rewritten_rows


def run(config: Dict[str, Any]) -> None:
    """
    Main entry point for the rewriting pipeline stage.

    This stage:
    1. Loads single-hop and multi-hop question datasets
    2. Sends each question to an LLM for rewriting
    3. Parses the rewritten questions
    4. Saves new datasets with rewritten questions
    """
    stage_cfg = config.get("pipeline", {}).get("rewriting", {})
    if not stage_cfg.get("run", False):
        logger.info("Rewriting stage is disabled. Skipping.")
        return

    logger.info("Starting question rewriting stage...")

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
            calls, indices = _build_rewriting_calls(
                single_hop_ds, QUESTION_REWRITING_SYSTEM_PROMPT, additional_instructions, is_multihop=False
            )

            if calls:
                responses = run_inference(config=config, step_name="rewriting", inference_calls=calls)

                rewritten_rows = _process_rewriting_responses(responses, indices, single_hop_ds)

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
            calls, indices = _build_rewriting_calls(
                multi_hop_ds, QUESTION_REWRITING_SYSTEM_PROMPT, additional_instructions, is_multihop=True
            )

            if calls:
                responses = run_inference(config=config, step_name="rewriting", inference_calls=calls)

                rewritten_rows = _process_rewriting_responses(responses, indices, multi_hop_ds)

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

    logger.success("Question rewriting stage completed")
