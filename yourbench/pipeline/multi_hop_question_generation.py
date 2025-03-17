# ============================================================
# multi_hop_question_generation.py
# ============================================================
"""
Author: @sumukshashidhar

Module Name:
------------
multi_hop_question_generation

Purpose:
--------
This module implements the multi-hop question generation stage within the YourBench pipeline. 
It processes a dataset of documents—each containing a list of multi-hop chunks—and generates 
multi-hop questions requiring integrative reasoning across those chunks. It uses a large 
language model to produce question-answer pairs in JSON format.

Usage:
------
This module is typically invoked as part of the overall YourBench pipeline. It expects:
1. A source dataset (e.g., documents with 'multihop_chunks' field).
2. Configuration for multi-hop question generation, such as sampling parameters and 
   additional instructions.
3. The pipeline orchestrator (in `handler.py`) calls `run(config)` if 
   `multi_hop_question_generation` is enabled in the YAML configuration.

The module then:
1. Optionally samples multi-hop chunks from each document.
2. Prompts a large language model to generate multi-hop question-answer pairs.
3. Parses and saves the generated questions in a structured dataset.

Error Handling and Logging:
---------------------------
- Comprehensive logging is performed using `loguru` at various levels to trace execution.
- Exceptions are caught and logged as errors, with the module attempting to continue 
  where practical.
- Critical issues produce warnings or errors and gracefully terminate the stage.

Module-Level Dependencies:
--------------------------
- Relies on the shared pipeline utilities (e.g., `yourbench.utils.dataset_engine`, 
  `yourbench.utils.inference_engine`, `yourbench.utils.prompts`).
- Preserves the existing signature and functionality for downstream consistency.
"""

import json
import re
import random
from dataclasses import dataclass, field
from typing import Dict, Any, List

from loguru import logger
from datasets import Dataset

from yourbench.utils.dataset_engine import (
    smart_load_dataset,
    smart_get_source_dataset_name,
    smart_get_output_dataset_name,
    smart_get_source_subset,
    smart_get_output_subset,
    save_dataset
)
from yourbench.utils.inference_engine import InferenceCall, run_inference
from yourbench.utils.prompts import (
    MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT,
    MULTI_HOP_QUESTION_GENERATION_USER_PROMPT,
)


@dataclass
class MultiHopQuestionRow:
    """
    Data structure to represent a single multi-hop question row.

    Attributes:
        document_id (str):
            Identifier of the document from which the question is generated.
        source_chunk_ids (List[str]):
            List of single-hop chunk IDs used in producing the multi-hop question.
        question (str):
            The generated multi-hop question text.
        self_answer (str):
            A plausible model-provided answer to the generated question.
        estimated_difficulty (int):
            Difficulty rating on a scale of 1 (easiest) to 10 (hardest).
        self_assessed_question_type (str):
            A descriptor for the question style (e.g., "analytical", "factual").
        generating_model (str):
            Name of the model that generated the question-answer pair.
        thought_process (str):
            Free-form text describing the model's reasoning for the question.
        citations (List[str]):
            Optional references/quotations from the combined chunks.
    """
    document_id: str
    source_chunk_ids: List[str]
    question: str
    self_answer: str
    estimated_difficulty: int
    self_assessed_question_type: str
    generating_model: str
    thought_process: str
    citations: List[str] = field(default_factory=list)


def run(config: Dict[str, Any]) -> None:
    """
    Execute the multi-hop question generation stage.

    This function orchestrates:
      1. Dataset loading (expecting 'multihop_chunks' in each row).
      2. Optional sampling of multi-hop chunks to manage cost.
      3. Sending prompt-based requests to a configured LLM for multi-hop Q&A generation.
      4. Parsing and structuring the results into a new dataset.

    Args:
        config (Dict[str, Any]): 
            The overall pipeline configuration dictionary, usually loaded 
            from a YAML file. Must include:
            - pipeline.multi_hop_question_generation.run (bool)
            - pipeline.multi_hop_question_generation.source_subset (str)
            - pipeline.multi_hop_question_generation.output_subset (str)
            - pipeline.multi_hop_question_generation.chunk_sampling (optional dict)
            - pipeline.multi_hop_question_generation.additional_instructions (str)

    Returns:
        None. The results are saved as a dataset (e.g., to disk or HF Hub),
        based on the configuration settings.

    Raises:
        Exception: Logs errors if something unexpected occurs during 
        inference or dataset processing, but does not halt the entire pipeline.
    """
    try:
        stage_cfg = config.get("pipeline", {}).get("multi_hop_question_generation", {})
        if not stage_cfg.get("run", False):
            logger.info("multi_hop_question_generation stage is disabled. Skipping.")
            return

        # Identify relevant dataset subsets
        source_dataset_name = smart_get_source_dataset_name("multi_hop_question_generation", config)
        source_subset = smart_get_source_subset("multi_hop_question_generation", config)
        output_dataset_name = smart_get_output_dataset_name("multi_hop_question_generation", config)
        output_subset = smart_get_output_subset("multi_hop_question_generation", config)

        logger.info("Loading dataset for multi-hop QG: '{}'", source_dataset_name)
        dataset = smart_load_dataset(source_dataset_name, config, source_subset)
        logger.info("Loaded dataset with {} rows.", len(dataset))

        # Prepare system message for LLM
        system_msg = {
            "role": "system",
            "content": MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT
        }

        all_inference_calls: List[InferenceCall] = []
        call_index_map: List[tuple] = []

        def sample_multi_hop_chunks(mh_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """
            Sample multi-hop chunks from each row based on config settings 
            to control cost (percentage or count-based).

            Args:
                mh_chunks (List[Dict[str, Any]]): 
                    List of multi-hop chunk dictionaries containing 'chunk_ids'
                    and 'chunks_text'.

            Returns:
                List[Dict[str, Any]]: 
                    The potentially sampled subset of multi-hop chunks.
            """
            chunk_sampling_cfg = stage_cfg.get("chunk_sampling", {})
            if not chunk_sampling_cfg:
                return mh_chunks

            mode = chunk_sampling_cfg.get("mode", "all").lower()
            value = chunk_sampling_cfg.get("value", 1.0)
            rand_seed = chunk_sampling_cfg.get("random_seed", 42)
            random.seed(rand_seed)

            total_multi_hops = len(mh_chunks)
            if total_multi_hops == 0:
                return mh_chunks

            if mode == "percentage":
                k = int(total_multi_hops * float(value))
                k = max(0, min(k, total_multi_hops))
                if k < total_multi_hops:
                    return random.sample(mh_chunks, k)
                return mh_chunks

            elif mode == "count":
                k = min(int(value), total_multi_hops)
                if k < total_multi_hops:
                    return random.sample(mh_chunks, k)
                return mh_chunks

            return mh_chunks

        # Build calls for each row
        for row_idx, row in enumerate(dataset):
            doc_summary = row.get("document_summary", "No summary provided.")
            title = row.get("document_filename", f"Document_{row_idx}")
            doc_id = row.get("document_id", f"doc_{row_idx}")

            multi_hop_chunks = row.get("multihop_chunks", [])
            if not isinstance(multi_hop_chunks, list) or not multi_hop_chunks:
                logger.debug(
                    "No multi-hop chunks found in row index={}, doc_id={}. Skipping row.",
                    row_idx, doc_id
                )
                continue

            chosen_multi_hops = sample_multi_hop_chunks(multi_hop_chunks)
            if not chosen_multi_hops:
                logger.debug(
                    "Row idx={} doc_id={} had multi-hop chunks but none after sampling.",
                    row_idx, doc_id
                )
                continue

            additional_instructions = stage_cfg.get("additional_instructions", "undergraduate")

            # For each multi-hop chunk, create an LLM prompt
            for mh_idx, mh_dict in enumerate(chosen_multi_hops):
                if not isinstance(mh_dict, dict):
                    continue

                subchunk_ids = mh_dict.get("chunk_ids", [])
                subchunk_texts = mh_dict.get("chunks_text", [])
                if not subchunk_texts:
                    logger.debug(
                        "Empty multi-hop chunk at row_idx={}, doc_id={}. Skipping.",
                        row_idx, doc_id
                    )
                    continue

                # Build the user prompt by enumerating each subchunk
                text_chunks_aggregated = ""
                for i, sc_text in enumerate(subchunk_texts):
                    text_chunks_aggregated += f"<text_chunk_{i}>{sc_text}</text_chunk_{i}>\n"

                user_prompt_str = MULTI_HOP_QUESTION_GENERATION_USER_PROMPT.format(
                    title=title,
                    document_summary=doc_summary,
                    chunks=text_chunks_aggregated,
                    additional_instructions=additional_instructions
                )
                user_msg = {"role": "user", "content": user_prompt_str}
                inference_call = InferenceCall(
                    messages=[system_msg, user_msg],
                    tags=["multi_hop_qa"]
                )
                all_inference_calls.append(inference_call)
                # Keep track of the row, doc_id, and subchunk_ids for reconstruction later
                call_index_map.append((row_idx, doc_id, subchunk_ids))

        # If no calls, exit
        if not all_inference_calls:
            logger.warning("No multi-hop inference calls were created. Exiting stage.")
            return

        logger.info("Sending {} multi-hop calls to inference...", len(all_inference_calls))
        responses_dict = run_inference(
            config=config,
            step_name="multi_hop_question_generation",
            inference_calls=all_inference_calls,
        )

        # Prepare final question rows
        final_multi_hop_questions: List[Dict[str, Any]] = []

        # Process each model that responded
        for model_name, model_responses in responses_dict.items():
            logger.info("Processing {} responses for model: {}", len(model_responses), model_name)
            if len(model_responses) != len(call_index_map):
                logger.error(
                    "Model '{}' returned {} responses, expected {}. Possible mismatch.",
                    model_name, len(model_responses), len(call_index_map)
                )

            for idx, raw_resp in enumerate(model_responses):
                if idx >= len(call_index_map):
                    break

                row_idx, doc_id, source_chunk_ids = call_index_map[idx]
                json_str = _extract_output_json(raw_resp)
                if not json_str.strip():
                    logger.warning(
                        "No parseable JSON for row={}, doc_id={} (model={}). Skipping.",
                        row_idx, doc_id, model_name
                    )
                    continue

                try:
                    question_answer_list = json.loads(json_str)
                except Exception as e:
                    logger.warning(
                        "Failed to parse JSON for row={}, doc_id={} (model={}): {}",
                        row_idx, doc_id, model_name, e
                    )
                    continue

                if not isinstance(question_answer_list, list):
                    logger.warning(
                        "Expected a list of question-answer pairs; got type={} instead. row={}, doc_id={}, model={}",
                        type(question_answer_list).__name__, row_idx, doc_id, model_name
                    )
                    continue

                # Construct final question data for each QA pair
                for qap in question_answer_list:
                    try:
                        question_text = qap.get("question", "").strip()
                        if not question_text:
                            logger.debug(
                                "Empty question found for row={}, doc_id={}, skipping pair.",
                                row_idx, doc_id
                            )
                            continue

                        self_answer = qap.get("answer", "").strip()
                        diff_raw = qap.get("estimated_difficulty", 5)
                        try:
                            diff_val = int(diff_raw)
                        except (ValueError, TypeError):
                            logger.warning(
                                "Invalid difficulty '{}' for doc_id={}, defaulting to 5",
                                diff_raw, doc_id
                            )
                            diff_val = 5

                        qtype = qap.get("question_type", "unknown")
                        if not isinstance(qtype, str):
                            qtype = str(qtype)

                        thought_process = qap.get("thought_process", "")
                        if not isinstance(thought_process, str):
                            thought_process = str(thought_process)

                        cits = qap.get("citations", [])
                        if not isinstance(cits, list):
                            logger.warning(
                                "Citations for doc_id={} is not a list. Converting to empty list.",
                                doc_id
                            )
                            cits = []

                        row_obj = MultiHopQuestionRow(
                            document_id=doc_id,
                            source_chunk_ids=source_chunk_ids,
                            question=question_text,
                            self_answer=self_answer,
                            estimated_difficulty=diff_val,
                            self_assessed_question_type=qtype,
                            generating_model=model_name,
                            thought_process=thought_process,
                            citations=cits
                        )
                        final_multi_hop_questions.append(row_obj.__dict__)

                    except Exception as pair_error:
                        logger.warning(
                            "Error processing QA pair for doc_id={}, skipping pair: {}",
                            doc_id, pair_error
                        )
                        continue

        if not final_multi_hop_questions:
            logger.warning("No valid multi-hop question rows produced. Exiting stage.")
            return

        logger.info(
            "Constructing multi-hop question dataset with {} rows...",
            len(final_multi_hop_questions)
        )

        # Convert to Hugging Face Dataset
        try:
            col_keys = list(final_multi_hop_questions[0].keys())
            dataset_dict = {k: [row[k] for row in final_multi_hop_questions] for k in col_keys}
            final_question_dataset = Dataset.from_dict(dataset_dict)
        except Exception as ds_error:
            logger.error("Failed to create dataset from multi-hop question rows: {}", ds_error)
            return

        # Save final dataset
        logger.info("Saving multi-hop question dataset as '{}'.", output_dataset_name)
        save_dataset(
            dataset=final_question_dataset,
            step_name="multi_hop_question_generation",
            config=config,
            output_dataset_name=output_dataset_name,
            output_subset=output_subset
        )
        logger.success("Multi-hop question generation completed successfully.")

    except Exception as outer_exc:
        logger.error("Error in multi_hop_question_generation run function: {}", str(outer_exc))
        logger.warning("Multi-hop question generation stage encountered errors.")


def _extract_tag_content(text: str, tag: str) -> str:
    """
    Extract content from a specified XML tag in a given text.

    Args:
        text (str): 
            The full string from which to extract content.
        tag (str): 
            The name of the XML tag to search for.

    Returns:
        str: The content found between <tag>...</tag>. 
             Returns an empty string if not found or parsing fails.
    """
    pattern = fr"<{tag}\s*>([\s\S]*?)</{tag}>"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _extract_output_json(raw_response: str) -> str:
    """
    Attempt to extract JSON from the model's raw response. 
    Looks for <output_json> blocks or fenced code blocks with ```json.

    Args:
        raw_response (str):
            The raw string returned from the model.

    Returns:
        str: A JSON string if found, or an empty string otherwise.
    """
    # Priority 1: <output_json> block
    extracted = _extract_tag_content(raw_response, "output_json")
    if extracted.strip():
        sanitized = _maybe_strip_triple_backticks(extracted)
        if sanitized.strip():
            return sanitized

    # Priority 2: ```json fenced code block
    fence_pattern = r"```json\s*([\s\S]*?)\s*```"
    fence_match = re.search(fence_pattern, raw_response)
    if fence_match:
        return fence_match.group(1).strip()

    # Priority 3: Best effort bracket-based extraction
    fallback_candidates = _best_effort_json_extract(raw_response)
    return fallback_candidates[0] if fallback_candidates else ""


def _maybe_strip_triple_backticks(text_in: str) -> str:
    """
    Remove triple backticks if the text is entirely enclosed in them.

    Args:
        text_in (str): The text potentially wrapped with triple backticks.

    Returns:
        str: The text without enclosing triple backticks, or the original text.
    """
    pattern = r"^\s*```(?:json)?\s*([\s\S]*?)\s*```$"
    m = re.match(pattern, text_in)
    if m:
        return m.group(1)
    return text_in


def _best_effort_json_extract(full_text: str) -> List[str]:
    """
    Attempt to locate JSON-like bracketed text from a larger string. 
    Collects all substring candidates starting with '{' or '[' and 
    ending with '}' or ']'.

    Args:
        full_text (str): 
            The raw text potentially containing JSON.

    Returns:
        List[str]: A list of bracket-delimited substrings that might be valid JSON.
    """
    pattern = r"([\[{].*?[\]}])"
    matches = re.findall(pattern, full_text, flags=re.DOTALL)
    candidates = []
    for match_text in matches:
        if (match_text.startswith("[") and match_text.endswith("]")) or \
           (match_text.startswith("{") and match_text.endswith("}")):
            candidates.append(match_text.strip())
    return candidates
