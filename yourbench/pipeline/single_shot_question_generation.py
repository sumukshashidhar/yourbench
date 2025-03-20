# ============================================================
# single_shot_question_generation.py
# ============================================================
"""
Author: @sumukshashidhar

This module implements the Single-Shot Question Generation stage of the YourBench pipeline.

Overview:
    - Given a dataset containing document summaries and their associated single-hop chunks,
      this stage generates question-answer pairs for each chunk using one or more LLMs.
    - The generated questions are intended to be standalone, moderately challenging,
      and reflect a deep understanding of the provided text chunk.

Usage:
    1) The pipeline will call the `run()` function from this module if the user configures
       `pipeline.single_shot_question_generation.run = True`.
    2) This function loads the required dataset (specified in the pipeline configuration),
       samples chunks if necessary, and calls an LLM to generate questions.
    3) The output is stored in a new dataset containing each generated question,
       an estimated difficulty rating, and the model's self-provided reasoning.

Stage-Specific Logging:
    - All errors and relevant log messages are recorded in `logs/single_shot_question_generation.log`.

Google-Style Docstrings:
    - This codebase uses Python type hints and Google-style docstrings for clarity,
      maintainability, and consistency.
"""

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import Dataset
from loguru import logger

from yourbench.utils.dataset_engine import (
    save_dataset,
    smart_get_output_dataset_name,
    smart_get_output_subset,
    smart_get_source_dataset_name,
    smart_get_source_subset,
    smart_load_dataset,
)
from yourbench.utils.inference_engine import InferenceCall, run_inference
from yourbench.utils.prompts import (
    QUESTION_GENERATION_SYSTEM_PROMPT,
    QUESTION_GENERATION_USER_PROMPT,
)


# Ensure the logs directory exists and set up a stage-specific log file.
os.makedirs("logs", exist_ok=True)
logger.add("logs/single_shot_question_generation.log", level="DEBUG")


@dataclass
class SingleHopQuestionRow:
    """
    Represents a single-hop question row derived from a single chunk of text.

    Attributes:
        chunk_id: A string identifier for the chunk from which this question was generated.
        document_id: Identifier for the parent document.
        question: The generated question text.
        self_answer: The LLM-produced short answer or reasoning.
        estimated_difficulty: An integer from 1-10 indicating the estimated difficulty.
        self_assessed_question_type: A descriptor for the type or style of question.
        generating_model: The model used to generate this question.
        thought_process: Free-form text describing how the question was derived or the
            model's chain-of-thought (if provided).
        raw_response: The full, unedited response from the model.
    """

    chunk_id: str
    document_id: str
    question: str
    self_answer: str
    estimated_difficulty: int
    self_assessed_question_type: str
    generating_model: str
    thought_process: str
    raw_response: str


def run(config: Dict[str, Any]) -> None:
    """
    Executes the Single-Shot Question Generation stage of the pipeline.

    This function performs the following steps:
      1. Checks configuration to ensure this stage is enabled.
      2. Loads the dataset containing document and chunk information.
      3. Optionally samples chunks from each document to control inference costs.
      4. Calls one or more LLMs to generate question-answer pairs for each sampled chunk.
      5. Parses the JSON from each LLM response and constructs a final question-level dataset.
      6. Saves the resulting dataset to local storage or the Hugging Face Hub, according to config.

    Args:
        config (Dict[str, Any]): The overall pipeline configuration dictionary.
            Expected keys and structure:
            - pipeline:
                - single_shot_question_generation:
                    run (bool): Whether to execute this stage.
                    source_subset (str): Subset of the dataset to load from.
                    output_subset (str): Subset of the dataset to save the results to.
                    additional_instructions (str): Extra prompts or instructions for the LLM.
                    chunk_sampling: Optional sub-dict specifying how to sample chunks.
            - model_roles and model_list: Model and inference configuration.

    Returns:
        None. Results are saved directly to a dataset (local or HF Hub).
    """
    stage_config = config.get("pipeline", {}).get("single_shot_question_generation", {})
    if not stage_config.get("run", False):
        logger.info("single_shot_question_generation stage is disabled. Skipping.")
        return

    # Identify dataset names and subsets
    source_dataset_name = smart_get_source_dataset_name("single_shot_question_generation", config)
    output_dataset_name = smart_get_output_dataset_name("single_shot_question_generation", config)
    source_subset = smart_get_source_subset("single_shot_question_generation", config)
    output_subset = smart_get_output_subset("single_shot_question_generation", config)

    logger.info("Loading chunked dataset for single-shot QG: {}", source_dataset_name)
    try:
        dataset = smart_load_dataset(source_dataset_name, config, dataset_subset=source_subset)
    except Exception as err:
        logger.error("Failed to load dataset '{0}' for single_shot_question_generation: {1}", source_dataset_name, err)
        return

    logger.info("Loaded dataset with {} rows.", len(dataset))

    # Prepare the system message for question generation
    system_message = {"role": "system", "content": QUESTION_GENERATION_SYSTEM_PROMPT}

    inference_calls = []
    call_index_mapping = []

    def sample_chunks_if_needed(chunks_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Samples chunks according to user configuration, either by percentage or count.
        Returns all chunks if no sampling configuration is provided.

        Args:
            chunks_list (List[Dict[str, Any]]): A list of chunk dictionaries with keys
                like "chunk_id" and "chunk_text".

        Returns:
            List[Dict[str, Any]]: A possibly reduced list of chunk dictionaries.
        """
        sampling_config = stage_config.get("chunk_sampling", {})
        if not sampling_config:
            return chunks_list

        mode = sampling_config.get("mode", "all").lower()
        value = sampling_config.get("value", 1.0)
        random_seed = sampling_config.get("random_seed", 42)
        random.seed(random_seed)

        total_chunks = len(chunks_list)
        if total_chunks == 0:
            return chunks_list

        # Handle sampling mode
        if mode == "percentage":
            # e.g., value = 0.5 => sample 50% of the chunks
            num_selected = int(total_chunks * float(value))
            num_selected = max(0, min(num_selected, total_chunks))
            if num_selected < total_chunks:
                return random.sample(chunks_list, num_selected)
            return chunks_list

        elif mode == "count":
            # e.g., value = 10 => sample 10 chunks
            num_selected = min(int(value), total_chunks)
            if num_selected < total_chunks:
                return random.sample(chunks_list, num_selected)
            return chunks_list

        return chunks_list

    # Create inference calls for each row (document)
    for row_index, row in enumerate(dataset):
        doc_summary = row.get("document_summary", "No summary available.")
        title = row.get("document_filename", f"Document_{row_index}")
        doc_id = row.get("document_id", f"doc_{row_index}")

        single_hop_chunks = row.get("chunks", [])
        if not isinstance(single_hop_chunks, list) or not single_hop_chunks:
            logger.debug("No chunks found in row index={} for doc_id={}. Skipping row.", row_index, doc_id)
            continue

        chosen_chunks = sample_chunks_if_needed(single_hop_chunks)
        additional_instructions = stage_config.get("additional_instructions", "undergraduate")

        # Build user messages for each chunk
        for chunk_index, chunk_info in enumerate(chosen_chunks):
            if not isinstance(chunk_info, dict):
                chunk_text = str(chunk_info)
                chunk_id = f"{doc_id}_{chunk_index}"
            else:
                chunk_text = chunk_info.get("chunk_text", "")
                chunk_id = chunk_info.get("chunk_id", f"{doc_id}_{chunk_index}")

            user_prompt_str = QUESTION_GENERATION_USER_PROMPT.format(
                title=title,
                document_summary=doc_summary,
                text_chunk=chunk_text,
                additional_instructions=additional_instructions,
            )
            user_message = {"role": "user", "content": user_prompt_str}

            inference_call = InferenceCall(messages=[system_message, user_message], tags=["single_shot_qa"])
            inference_calls.append(inference_call)
            call_index_mapping.append((row_index, doc_id, chunk_id))

    if not inference_calls:
        logger.warning("No inference calls were created for single_shot_question_generation.")
        return

    logger.info("Sending {} calls to inference for single-shot question generation.", len(inference_calls))
    try:
        responses_dict = run_inference(
            config=config,
            step_name="single_shot_question_generation",
            inference_calls=inference_calls,
        )
    except Exception as err:
        logger.error("Inference failed for single_shot_question_generation: {}", err)
        return

    # Container for final question dataset rows
    question_dataset_rows: List[Dict[str, Any]] = []

    # Process the responses
    for model_name, model_responses in responses_dict.items():
        logger.info("Processing {} responses from model: {}", len(model_responses), model_name)

        if len(model_responses) != len(call_index_mapping):
            logger.error(
                "Model '{}' returned {} responses but we have {} calls. Possible mismatch.",
                model_name,
                len(model_responses),
                len(call_index_mapping),
            )

        for idx, raw_response in enumerate(model_responses):
            if idx >= len(call_index_mapping):
                break

            row_index, doc_id, chunk_id = call_index_mapping[idx]

            json_text = _extract_output_json(raw_response)
            if not json_text.strip():
                logger.warning(
                    "No parseable JSON found for row_index={}, chunk_id={}, model={}. Skipping.",
                    row_index,
                    chunk_id,
                    model_name,
                )
                continue

            try:
                question_answer_pairs = json.loads(json_text)
            except Exception as parse_err:
                logger.warning(
                    "Failed to parse JSON for row_index={}, chunk_id={}, model={}: {}",
                    row_index,
                    chunk_id,
                    model_name,
                    parse_err,
                )
                continue

            if not isinstance(question_answer_pairs, list):
                logger.warning(
                    "Expected a list of QA pairs, got something else for row_index={}, chunk_id={}, model={}.",
                    row_index,
                    chunk_id,
                    model_name,
                )
                continue

            # Process each QA pair
            for pair in question_answer_pairs:
                if not isinstance(pair, dict):
                    logger.warning(
                        "Invalid QA pair structure at row_index={}, chunk_id={}, model={}. Expected dict, got {}",
                        row_index,
                        chunk_id,
                        model_name,
                        type(pair).__name__,
                    )
                    continue

                question_text = pair.get("question") or ""
                question_text = question_text.strip() if isinstance(question_text, str) else str(question_text).strip()
                if not question_text:
                    logger.debug("Empty question found; skipping. row_index={}, chunk_id={}", row_index, chunk_id)
                    continue

                # Handle potential non-string answers
                answer_raw = pair.get("answer", "")
                self_answer = answer_raw.strip() if isinstance(answer_raw, str) else str(answer_raw).strip()

                # Handle potential non-int difficulty
                difficulty_raw = pair.get("estimated_difficulty", 5)
                try:
                    difficulty_val = int(difficulty_raw)
                except (ValueError, TypeError):
                    logger.warning("Invalid estimated_difficulty '{}', defaulting to 5", difficulty_raw)
                    difficulty_val = 5
                # Ensure difficulty is in range 1-10
                difficulty_val = max(1, min(10, difficulty_val))

                # Ensure question_type is a string
                question_type = pair.get("question_type", "unknown")
                if not isinstance(question_type, str):
                    question_type = str(question_type)

                # Ensure thought_process is a string
                thought_process = pair.get("thought_process", "")
                if not isinstance(thought_process, str):
                    thought_process = str(thought_process)

                # Construct final data row
                question_row = SingleHopQuestionRow(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    question=question_text,
                    self_answer=self_answer,
                    estimated_difficulty=difficulty_val,
                    self_assessed_question_type=question_type,
                    generating_model=model_name,
                    thought_process=thought_process,
                    raw_response=raw_response,
                )
                question_dataset_rows.append(question_row.__dict__)

    if not question_dataset_rows:
        logger.warning("No valid questions produced in single_shot_question_generation.")
        return

    logger.info("Constructing final dataset with {} single-hop questions.", len(question_dataset_rows))
    try:
        column_names = list(question_dataset_rows[0].keys())
    except IndexError:
        logger.error("No question rows available to generate dataset columns. Exiting.")
        return

    # Convert to HF Dataset
    final_data = {column: [row[column] for row in question_dataset_rows] for column in column_names}
    question_dataset = Dataset.from_dict(final_data)

    # Save the dataset
    logger.info("Saving single-shot questions to dataset '{}'.", output_dataset_name)
    try:
        save_dataset(
            dataset=question_dataset,
            config=config,
            step_name="single_shot_question_generation",
            output_dataset_name=output_dataset_name,
            output_subset=output_subset,
        )
        logger.success("Single-shot question generation completed successfully.")
    except Exception as save_err:
        logger.error("Error saving single-shot question dataset: {}", save_err)


def _extract_tag_content(text: str, tag: str) -> str:
    """
    Extracts text enclosed in the specified XML tag from a given string.

    Args:
        text (str): The source string potentially containing XML tags.
        tag (str): The name of the XML tag to extract content from.

    Returns:
        str: The extracted content within <tag> ... </tag>, or empty if none found.
    """
    if not text or not isinstance(text, str):
        return ""

    try:
        pattern = rf"<{tag}\s*>([\s\S]*?)</{tag}>"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    except Exception as e:
        logger.debug("Error extracting tag content for tag '{}': {}", tag, e)
    return ""


def _extract_output_json(raw_response: str) -> str:
    """
    Extracts JSON content from the model response by searching
    for <output_json> blocks or fenced code blocks with `json`.

    Args:
        raw_response (str): The raw string returned by the model.

    Returns:
        str: The JSON content as a string, or an empty string if not found.
    """
    if not raw_response or not isinstance(raw_response, str):
        return ""

    try:
        # Check for <output_json> block first
        extracted = _extract_tag_content(raw_response, "output_json")
        if extracted and extracted.strip():
            sanitized = _maybe_strip_triple_backticks(extracted)
            if sanitized and sanitized.strip():
                return sanitized

        # Check for ```json fenced code block
        fenced_pattern = r"```json\s*([\s\S]*?)\s*```"
        fenced_match = re.search(fenced_pattern, raw_response)
        if fenced_match:
            return fenced_match.group(1).strip()

        # Fallback bracket-based extraction
        fallback_candidates = _best_effort_json_extract(raw_response)
        return fallback_candidates[0] if fallback_candidates else ""
    except Exception as e:
        logger.debug("Error extracting JSON from response: {}", e)
        return ""


def _maybe_strip_triple_backticks(text_in: str) -> str:
    """
    Removes triple backticks (``` or ```json) from the beginning and end of a string,
    if present.

    Args:
        text_in (str): The string that may be wrapped in triple backticks.

    Returns:
        str: The unwrapped text or the original string if no wrapping was found.
    """
    if not text_in or not isinstance(text_in, str):
        return ""

    try:
        pattern = r"^\s*```(?:json)?\s*([\s\S]*?)\s*```$"
        match = re.match(pattern, text_in)
        if match:
            return match.group(1)
    except Exception as e:
        logger.debug("Error stripping backticks: {}", e)
    return text_in


def _best_effort_json_extract(full_text: str) -> List[str]:
    """
    Searches for bracket-delimited content (curly or square) that might be valid JSON.
    Returns a list of candidate strings.

    Args:
        full_text (str): The complete text from which to extract JSON-like content.

    Returns:
        List[str]: A list of candidate JSON strings. May be empty if none found.
    """
    if not full_text or not isinstance(full_text, str):
        return []

    candidates = []
    try:
        pattern = r"([\[{].*?[\]}])"
        matches = re.findall(pattern, full_text, flags=re.DOTALL)
        for match_str in matches:
            if (match_str.startswith("[") and match_str.endswith("]")) or (
                match_str.startswith("{") and match_str.endswith("}")
            ):
                candidates.append(match_str.strip())
    except Exception as e:
        logger.debug("Error in best effort JSON extraction: {}", e)
    return candidates
