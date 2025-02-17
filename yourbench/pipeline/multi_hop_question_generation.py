# yourbench/pipeline/multi_hop_question_generation.py

"""
Multi-Hop Question Generation Module

This updated version produces a dataset where each row corresponds to a single
multi-hop question, analogous to the single-shot question generation scenario.

Key Changes:
------------
1. For each row in the chunked dataset, we iterate over each 'multihop_chunks' item 
   separately instead of aggregating them all into a single prompt per row.
2. We generate one InferenceCall per (row, multi-hop-chunk) pair.
3. After inference, we parse each model's response into question-level rows, yielding 
   a final dataset with one row per question (instead of a single column containing 
   lists of Q&A).
4. The output columns mirror single-shot question generation fields:
   [
   "chunk_id", 
   "document_id", 
   "chunk_location_id", 
   "diversification_seed", 
   "question", 
   "self_answer", 
   "estimated_difficulty", 
   "self_assessed_question_type", 
   "generating_model"
   ]

5. We have added more robust JSON parsing to gracefully handle malformed responses
   from the model. This prevents the entire multi-hop generation from failing
   when only a subset of the JSON is properly formatted.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List

from loguru import logger
from datasets import Dataset

from yourbench.utils.dataset_engine import smart_load_dataset
from yourbench.utils.saving_engine import save_dataset
from yourbench.utils.inference_engine import InferenceCall, run_inference
from yourbench.utils.prompts import (
    MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT,
    MULTI_HOP_QUESTION_GENERATION_USER_PROMPT,
)

@dataclass
class MultiHopQuestionRow:
    """
    Data structure describing a single multi-hop question as one row in the final dataset.
    """
    chunk_id: str
    document_id: str
    chunk_location_id: int
    diversification_seed: str
    question: str
    self_answer: str
    estimated_difficulty: int
    self_assessed_question_type: str
    generating_model: str


def run(config: Dict[str, Any]) -> None:
    """
    Main entry point for multi-hop question generation.

    Steps:
      1. Load the chunked dataset from config["pipeline"]["multi_hop_question_generation"]["source_dataset_name"].
      2. Build inference calls for each row-chunk pair in 'multihop_chunks'. 
         (We record row_idx and chunk_idx for reconstruction.)
      3. Call run_inference(...) to handle all calls in parallel for all specified models.
      4. Parse each model's responses into a question-level list of dictionaries, with one row per question:
            {
              "chunk_id", 
              "document_id", 
              "chunk_location_id", 
              "diversification_seed", 
              "question", 
              "self_answer", 
              "estimated_difficulty", 
              "self_assessed_question_type", 
              "generating_model"
            }
      5. Convert that list into a Hugging Face Dataset, then use save_dataset(...) 
         so we can optionally push to the hub.

    Config Example:
      pipeline:
        multi_hop_question_generation:
          source_dataset_name: yb_demo_chunked_documents
          output_dataset_name: yb_demo_multi_hop_questions
          local_dataset_path: data/example/multi_hop_questions
          concat_existing_dataset: false
          run: true

      model_roles:
        multi_hop_question_generation:
          - some_model_name
          - another_model_name
    """

    stage_cfg = config.get("pipeline", {}).get("multi_hop_question_generation", {})
    if not stage_cfg.get("run", False):
        logger.info("multi_hop_question_generation stage is disabled. Skipping.")
        return

    source_dataset_name = stage_cfg["source_dataset_name"]
    output_dataset_name = stage_cfg["output_dataset_name"]
    logger.info("Loading chunked dataset: {}", source_dataset_name)

    dataset = smart_load_dataset(source_dataset_name, config)
    logger.info("Loaded dataset with {} rows.", len(dataset))

    # Prepare the system prompt once
    system_message = {"role": "system", "content": MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT}

    all_inference_calls: List[InferenceCall] = []
    call_index_to_row_chunk: List[tuple] = []

    diversification_seed = stage_cfg.get("diversification_seed", "multihop_seed")

    for row_idx, row in enumerate(dataset):
        doc_summary = row.get("document_summary", "No summary provided.")
        title = row.get("document_filename", f"Document_{row_idx}")

        multi_hop_chunks = row.get("multihop_chunks", [])
        if not multi_hop_chunks:
            continue

        for c_idx, chunk_text in enumerate(multi_hop_chunks):
            chunk_section = f"<text_chunk_0>{chunk_text}</text_chunk_0>"
            test_audience = stage_cfg.get("test_audience", "undergraduate")

            user_prompt_filled = MULTI_HOP_QUESTION_GENERATION_USER_PROMPT.format(
                title=title,
                document_summary=doc_summary,
                chunks=chunk_section,
                test_audience=test_audience
            )

            user_message = {"role": "user", "content": user_prompt_filled}
            inference_call = InferenceCall(
                messages=[system_message, user_message],
                tags=["multi_hop_qa"]
            )
            all_inference_calls.append(inference_call)
            call_index_to_row_chunk.append((row_idx, c_idx))

    if not all_inference_calls:
        logger.warning("No multi-hop chunks found. Exiting multi-hop question generation.")
        return

    logger.info("Sending {} total calls to inference for multi-hop question generation.", len(all_inference_calls))

    responses_dict = run_inference(
        config=config,
        step_name="multi_hop_question_generation",
        inference_calls=all_inference_calls,
    )

    question_dataset_rows: List[Dict[str, Any]] = []

    for model_name, model_responses in responses_dict.items():
        logger.info("Processing {} responses for model: {}", len(model_responses), model_name)

        if len(model_responses) != len(call_index_to_row_chunk):
            logger.error(
                "Model '{}' returned {} responses, but we expected {}. Some mismatch occurred.",
                model_name, len(model_responses), len(call_index_to_row_chunk)
            )

        for call_idx, raw_response in enumerate(model_responses):
            if call_idx >= len(call_index_to_row_chunk):
                break

            row_idx, c_idx = call_index_to_row_chunk[call_idx]
            doc_id = dataset[row_idx].get("document_id", f"doc_{row_idx}")

            # === Attempt to extract JSON content from the raw_response ===
            parsed_json_str = _extract_output_json(raw_response)
            if not parsed_json_str.strip():
                logger.warning(
                    "No <output_json> block found for row {}, chunk {}. Model={}",
                    row_idx, c_idx, model_name
                )
                # We'll try a fallback approach if there's leftover text that might be JSON
                fallback_json_list = _best_effort_json_extract(raw_response)
                if fallback_json_list:
                    # Attempt to parse if any bracket-block is found
                    for candidate_json in fallback_json_list:
                        try:
                            question_answer_pairs = json.loads(candidate_json)
                            # If successful, we proceed
                            _parse_and_append_rows(
                                question_answer_pairs,
                                row_idx,
                                c_idx,
                                doc_id,
                                model_name,
                                diversification_seed,
                                question_dataset_rows
                            )
                            # No need to parse further candidates if one is successful
                            break
                        except Exception:
                            # Just continue to next candidate
                            pass
                continue

            try:
                question_answer_pairs = json.loads(parsed_json_str)
                # If we parse successfully, proceed to build rows
                _parse_and_append_rows(
                    question_answer_pairs,
                    row_idx,
                    c_idx,
                    doc_id,
                    model_name,
                    diversification_seed,
                    question_dataset_rows
                )
            except Exception as e:
                logger.warning(
                    "Failed to parse JSON for row {}, chunk {} (model={}): {}",
                    row_idx, c_idx, model_name, e
                )
                # As a fallback, we attempt best-effort extraction of bracketed JSON
                fallback_json_list = _best_effort_json_extract(raw_response)
                for candidate_json in fallback_json_list:
                    try:
                        question_answer_pairs = json.loads(candidate_json)
                        _parse_and_append_rows(
                            question_answer_pairs,
                            row_idx,
                            c_idx,
                            doc_id,
                            model_name,
                            diversification_seed,
                            question_dataset_rows
                        )
                        break  # Only keep the first successful parse
                    except Exception:
                        pass

    if not question_dataset_rows:
        logger.warning("No valid question rows produced from multi-hop generation. Exiting.")
        return

    logger.info("Constructing multi-hop question-level dataset with {} rows...", len(question_dataset_rows))

    question_dataset = Dataset.from_dict({
        k: [d[k] for d in question_dataset_rows]
        for k in question_dataset_rows[0].keys()
    })

    logger.info("Saving multi-hop question dataset to HF Hub under '{}'.", output_dataset_name)
    save_dataset(
        dataset=question_dataset,
        step_name="multi_hop_question_generation",
        config=config,
        output_dataset_name=output_dataset_name
    )
    logger.success("Multi-hop question generation completed successfully.")


def _extract_tag_content(text: str, tag: str) -> str:
    pattern = fr"<{tag}\s*>([\s\S]*?)</{tag}>"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""

### ADDED OR MODIFIED CODE HERE ###
def _extract_output_json(raw_response: str) -> str:
    """
    Attempt to extract JSON from either <output_json>...</output_json> or
    from a triple-backtick fenced code block: ```json ... ```.

    Returns the raw JSON string if found, otherwise empty string.
    """
    # 1. Try <output_json> first
    extracted = _extract_tag_content(raw_response, "output_json")
    if extracted.strip():
        return extracted

    # 2. If no <output_json>, look for ```json fenced content
    fence_pattern = r"```json\s*([\s\S]*?)\s*```"
    fence_match = re.search(fence_pattern, raw_response)
    if fence_match:
        return fence_match.group(1).strip()

    # 3. Nothing found
    return ""


### ADDED OR MODIFIED CODE HERE ###
def _best_effort_json_extract(full_text: str) -> List[str]:
    """
    When the direct <output_json> or triple-backtick parse fails, we try a crude
    fallback approach:
      - Find all substrings that start with '[' or '{' and end with ']' or '}'
      - Return them as candidate JSON blocks, letting the caller attempt json.loads(...)
      - This approach picks up partial or extra text that might still yield valid JSON

    Returns a list of candidate JSON snippet strings (possibly empty).
    """
    candidates = []
    # Simple pattern that tries to find bracketed sections in the text
    # that might be valid JSON
    pattern = r"([\[{].*?[\]}])"
    matches = re.findall(pattern, full_text, flags=re.DOTALL)
    for m in matches:
        snippet = m.strip()
        if (snippet.startswith("[") and snippet.endswith("]")) or \
           (snippet.startswith("{") and snippet.endswith("}")):
            candidates.append(snippet)
    return candidates


### ADDED OR MODIFIED CODE HERE ###
def _parse_and_append_rows(
    question_answer_pairs: Any,
    row_idx: int,
    c_idx: int,
    doc_id: str,
    model_name: str,
    diversification_seed: str,
    question_dataset_rows: List[Dict[str, Any]]
) -> None:
    """
    Given a loaded JSON (list of question dictionaries), parse them and
    append them to question_dataset_rows in the required structure.
    This is a helper to avoid repeated code.
    """
    if not isinstance(question_answer_pairs, list):
        # We expect a list of objects
        return
    for qap in question_answer_pairs:
        question_text = qap.get("question", "")
        self_answer_text = qap.get("thought_process", "")
        difficulty = qap.get("estimated_difficulty", 5)
        question_type = qap.get("question_type", "unknown")

        chunk_id = f"row_{row_idx}_multihop_{c_idx}"

        question_row = MultiHopQuestionRow(
            chunk_id=chunk_id,
            document_id=doc_id,
            chunk_location_id=c_idx,
            diversification_seed=diversification_seed,
            question=question_text,
            self_answer=self_answer_text,
            estimated_difficulty=difficulty,
            self_assessed_question_type=question_type,
            generating_model=model_name,
        )

        question_dataset_rows.append(question_row.__dict__)
