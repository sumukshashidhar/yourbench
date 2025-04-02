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

import random
from typing import Any
from dataclasses import field, dataclass

from loguru import logger

from datasets import Dataset
from yourbench.utils.prompts import (
    QUESTION_GENERATION_USER_PROMPT,
    QUESTION_GENERATION_SYSTEM_PROMPT,
)
from yourbench.utils.dataset_engine import (
    custom_load_dataset,
    custom_save_dataset,
)

# Import the unified parsing function
from yourbench.utils.parsing_engine import parse_qa_pairs_from_response
from yourbench.utils.inference_engine import InferenceCall, run_inference


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
        thought_process: Free-form text describing how the question was derived.
        raw_response: The full, unedited response from the model.
        citations: A list of references or quotes extracted from the chunk.
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
    citations: list[str]


@dataclass
class ChunkSamplingConfig:
    mode: str = "all"
    value: float = 1.0
    random_seed: int = 42


@dataclass
class SingleShotQuestionGenerationConfig:
    run: bool = False
    source_subset: str = ""
    output_subset: str = ""
    additional_instructions: str = "Generate questions to test an undergraduate student"
    chunk_sampling: ChunkSamplingConfig = field(default_factory=ChunkSamplingConfig)


@dataclass
class DocumentRow:
    document_summary: str = "No summary available."
    document_filename: str = ""
    document_id: str = ""
    chunks: list[dict[str, Any]] = field(default_factory=list)


def run(config: dict[str, Any]) -> None:
    """
    Executes the Single-Shot Question Generation stage of the pipeline.
    """
    stage_config = _load_stage_config(config)
    if not stage_config.run:
        logger.info("single_shot_question_generation stage is disabled. Skipping.")
        return

    dataset = custom_load_dataset(config=config, subset="chunked")
    logger.info(f"Loaded chunked subset with {len(dataset)} rows for Single-shot question generation.")

    inference_calls, call_index_mapping = _build_inference_calls(dataset, stage_config)
    if not inference_calls:
        logger.warning("No inference calls were created for single_shot_question_generation.")
        return

    responses_dict = _execute_inference(inference_calls, config)
    if not responses_dict:
        return

    question_dataset = _process_responses_and_build_dataset(responses_dict, call_index_mapping)
    if question_dataset is None or len(question_dataset) == 0:
        logger.warning("No valid questions produced in single_shot_question_generation.")
        return

    custom_save_dataset(dataset=question_dataset, config=config, subset="single_shot_questions")
    logger.success("Single-shot question generation completed successfully.")


def _load_stage_config(config: dict[str, Any]) -> SingleShotQuestionGenerationConfig:
    """
    Extract the stage-specific configuration from the pipeline config.
    """
    pipeline_config = config.get("pipeline", {})
    stage_config_dict = pipeline_config.get("single_shot_question_generation", {})
    chunk_sampling_cfg = stage_config_dict.get("chunk_sampling", {})

    # For readability: if len(chunk_sampling_cfg) == 0
    if len(chunk_sampling_cfg) == 0:
        chunk_sampling = ChunkSamplingConfig()
    else:
        chunk_sampling = ChunkSamplingConfig(
            mode=chunk_sampling_cfg.get("mode", "all"),
            value=chunk_sampling_cfg.get("value", 1.0),
            random_seed=chunk_sampling_cfg.get("random_seed", 42),
        )

    return SingleShotQuestionGenerationConfig(
        run=stage_config_dict.get("run", False),
        source_subset=stage_config_dict.get("source_subset", ""),
        output_subset=stage_config_dict.get("output_subset", ""),
        additional_instructions=stage_config_dict.get("additional_instructions", "undergraduate"),
        chunk_sampling=chunk_sampling,
    )


def _sample_chunks_if_needed(
    chunks_list: list[dict[str, Any]], chunk_sampling: ChunkSamplingConfig
) -> list[dict[str, Any]]:
    """
    Samples chunks according to user configuration, either by percentage or count.
    Returns all chunks if no sampling configuration is provided or invalid.
    """
    if not chunks_list:
        return chunks_list

    mode = chunk_sampling.mode.lower()
    value = chunk_sampling.value
    random_seed = chunk_sampling.random_seed
    random.seed(random_seed)

    total_chunks = len(chunks_list)
    if total_chunks == 0:
        return chunks_list

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

    # "all" or unrecognized mode => return all
    return chunks_list


def _build_inference_calls(dataset, stage_config: SingleShotQuestionGenerationConfig):
    """
    Create the InferenceCall objects needed for single-shot question generation.
    Returns the list of calls and a parallel mapping of (row_index, doc_id, chunk_id).
    """
    system_message = {"role": "system", "content": QUESTION_GENERATION_SYSTEM_PROMPT}
    inference_calls = []
    call_index_mapping = []

    for row_index, row in enumerate(dataset):
        doc_row = DocumentRow(
            document_summary=row.get("document_summary", "No summary available."),
            document_filename=row.get("document_filename", f"Document_{row_index}"),
            document_id=row.get("document_id", f"doc_{row_index}"),
            chunks=row.get("chunks", []),
        )

        single_hop_chunks = doc_row.chunks
        if not isinstance(single_hop_chunks, list) or not single_hop_chunks:
            logger.debug(f"No chunks found in row index={row_index} for doc_id={doc_row.document_id}. Skipping row.")
            continue

        chosen_chunks = _sample_chunks_if_needed(single_hop_chunks, stage_config.chunk_sampling)
        additional_instructions = stage_config.additional_instructions

        # Build user messages for each chunk
        for chunk_index, chunk_info in enumerate(chosen_chunks):
            if not isinstance(chunk_info, dict):
                chunk_text = str(chunk_info)
                chunk_id = f"{doc_row.document_id}_{chunk_index}"
            else:
                chunk_text = chunk_info.get("chunk_text", "")
                chunk_id = chunk_info.get("chunk_id", f"{doc_row.document_id}_{chunk_index}")

            user_prompt_str = QUESTION_GENERATION_USER_PROMPT.format(
                title=doc_row.document_filename,
                document_summary=doc_row.document_summary,
                text_chunk=chunk_text,
                additional_instructions=additional_instructions,
            )
            user_message = {"role": "user", "content": user_prompt_str}

            inference_call = InferenceCall(messages=[system_message, user_message], tags=["single_shot_qa"])
            inference_calls.append(inference_call)
            call_index_mapping.append((row_index, doc_row.document_id, chunk_id))

    return inference_calls, call_index_mapping


def _execute_inference(inference_calls, config: dict[str, Any]):
    """
    Sends the prepared inference calls to the LLM(s). Returns a dict of responses.
    """
    logger.info(f"Sending {len(inference_calls)} calls to inference for single-shot question generation.")
    try:
        return run_inference(
            config=config,
            step_name="single_shot_question_generation",
            inference_calls=inference_calls,
        )
    except Exception as err:
        logger.error(f"Inference failed for single_shot_question_generation: {err}")
        return {}


def _process_responses_and_build_dataset(
    responses_dict: dict[str, list[str]], call_index_mapping: list[tuple]
) -> Dataset:
    """
    Take the LLM responses, parse them, and build a Hugging Face Dataset
    of single-shot question rows.
    """
    question_dataset_rows = []

    for model_name, model_responses in responses_dict.items():
        logger.info(f"Processing {len(model_responses)} responses from model: {model_name}")
        if len(model_responses) != len(call_index_mapping):
            logger.error(
                f"Model '{model_name}' returned {len(model_responses)} responses but expected {len(call_index_mapping)}. Mismatch."
            )

        for idx, raw_response in enumerate(model_responses):
            if idx >= len(call_index_mapping):
                break

            row_index, doc_id, chunk_id = call_index_mapping[idx]
            qa_pairs = parse_qa_pairs_from_response(raw_response)

            # If parsing fails or returns nothing, still create a fallback row
            if not qa_pairs:
                logger.warning(
                    f"No parseable JSON found (or empty list) for row_index={row_index}, chunk_id={chunk_id}, model={model_name}. Creating fallback row."
                )
                continue

            # Otherwise, process each QA pair
            for pair in qa_pairs:
                try:
                    # Safely extract data from pair
                    question_text = str(pair.get("question", "")).strip()
                    answer_text = str(pair.get("answer", "")).strip()
                    difficulty_val = _force_int_in_range(pair.get("estimated_difficulty", 5), 1, 10)
                    question_type = str(pair.get("question_type", "unknown"))
                    thought_process = str(pair.get("thought_process", ""))
                    citations = pair.get("citations", [])
                    if not isinstance(citations, list):
                        citations = []

                    if not question_text:
                        logger.debug(f"Empty question found; skipping this QA pair (row_index={row_index}).")
                        continue

                    # Build final row
                    question_row = SingleHopQuestionRow(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        question=question_text,
                        self_answer=answer_text,
                        estimated_difficulty=difficulty_val,
                        self_assessed_question_type=question_type,
                        generating_model=model_name,
                        thought_process=thought_process,
                        raw_response=raw_response,
                        citations=citations,
                    )
                    question_dataset_rows.append(question_row.__dict__)
                except Exception as e:
                    logger.error(f"Error processing QA pair for row_index={row_index}, chunk_id={chunk_id}: {e}")
                    continue

    if not question_dataset_rows:
        return None

    logger.info(f"Constructing final dataset with {len(question_dataset_rows)} single-hop questions.")
    column_names = list(question_dataset_rows[0].keys())
    final_data = {column: [row[column] for row in question_dataset_rows] for column in column_names}
    return Dataset.from_dict(final_data)


def _force_int_in_range(value: Any, min_val: int, max_val: int) -> int:
    """
    Convert a value to int and clamp it between min_val and max_val.
    """
    try:
        ivalue = int(value)
    except (ValueError, TypeError):
        ivalue = (min_val + max_val) // 2
    return max(min_val, min(ivalue, max_val))
