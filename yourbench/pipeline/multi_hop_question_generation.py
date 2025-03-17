# ============================================================
# multi_hop_question_generation.py
# ============================================================
"""
Multi-Hop Question Generation Module

Minimal approach:
----------------
- Each row in the dataset has "multihop_chunks", which is a list of dicts:
    { "chunk_ids": [...], "chunks_text": [...] }
- For each such multi-hop item, we generate questions that may require
  combining information across multiple single-hop chunks. 
- In the final question dataset, we store a list of source_chunk_ids to 
  show which single-hop chunks were used. We do not store chunk_uuid or 
  location indices. The final row is basically:
    {
      "document_id": ...,
      "source_chunk_ids": [...],
      "question": ...,
      "self_answer": ...,
      ...
    }
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
    Minimal structure for multi-hop question rows:
      - document_id: which doc
      - source_chunk_ids: which single-hop chunks are used
      - question: the generated question
      - self_answer: the model's best guess or reasoning
      - estimated_difficulty: integer from 1-10
      - self_assessed_question_type: a descriptor for question style or category
      - generating_model: which model generated the question
      - thought_process: free-form text describing how the question was derived
      - citations: optional list of references or quotes from the combined chunks
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
    Main entry point for multi-hop question generation. 

    Steps:
      1) Load the dataset, which must have a column "multihop_chunks": a list of dicts.
      2) Optionally sample a subset of these multi-hop chunks if config requests it 
         (to control cost).
      3) For each multi-hop chunk, call an LLM to produce multiple multi-hop Q&A pairs.
      4) Parse and store them in a question-level dataset with minimal fields.

    The YAML config block might look like:

    multi_hop_question_generation:
      run: true
      source_subset: chunked_documents
      output_subset: multi_hop_questions
      additional_instructions: "Generate advanced integrative questions"
      chunk_sampling:
        mode: "count"      # or "percentage"
        value: 10
        random_seed: 123

    If chunk_sampling is omitted, we generate from all available multi-hop chunks.
    """
    try:
        # === Retrieve stage config from the pipeline dict ===
        stage_cfg = config.get("pipeline", {}).get("multi_hop_question_generation", {})
        if not stage_cfg.get("run", False):
            logger.info("multi_hop_question_generation stage is disabled. Skipping.")
            return

        source_dataset_name = smart_get_source_dataset_name("multi_hop_question_generation", config)
        source_subset = smart_get_source_subset("multi_hop_question_generation", config)
        output_dataset_name = smart_get_output_dataset_name("multi_hop_question_generation", config)
        output_subset = smart_get_output_subset("multi_hop_question_generation", config)

        logger.info("Loading dataset for multi-hop QG: '{}'", source_dataset_name)
        dataset = smart_load_dataset(source_dataset_name, config, source_subset)
        logger.info("Loaded dataset with {} rows.", len(dataset))

        # === System-level prompt for multi-hop question generation ===
        system_msg = {"role": "system", "content": MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT}

        all_inference_calls: List[InferenceCall] = []
        call_index_map: List[tuple] = []

        # === Helper function to sample multi-hop chunks if needed ===
        def sample_multihop_if_needed(mh_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """
            If chunk_sampling is specified, sample a fraction or fixed number 
            of multi-hop chunks from the provided list. Otherwise, return all.
            """
            chunk_sampling_cfg = stage_cfg.get("chunk_sampling", {})
            if not chunk_sampling_cfg:
                return mh_chunks

            mode = chunk_sampling_cfg.get("mode", "all").lower()  # 'percentage' or 'count'
            value = chunk_sampling_cfg.get("value", 1.0)
            rand_seed = chunk_sampling_cfg.get("random_seed", 42)
            random.seed(rand_seed)

            total_mh = len(mh_chunks)
            if total_mh == 0:
                return mh_chunks

            if mode == "percentage":
                k = int(total_mh * float(value))
                k = max(0, min(k, total_mh))
                if k < total_mh:
                    return random.sample(mh_chunks, k)
                return mh_chunks

            elif mode == "count":
                k = min(int(value), total_mh)
                if k < total_mh:
                    return random.sample(mh_chunks, k)
                return mh_chunks

            return mh_chunks

        # === Build calls for each row (and each multi-hop chunk in that row) ===
        for row_idx, row in enumerate(dataset):
            doc_summary = row.get("document_summary", "No summary provided.")
            title = row.get("document_filename", f"Document_{row_idx}")
            doc_id = row.get("document_id", f"doc_{row_idx}")

            multi_hop_chunks = row.get("multihop_chunks", [])
            if not isinstance(multi_hop_chunks, list) or not multi_hop_chunks:
                logger.debug("No multi-hop chunks found in row index={}, doc_id={}. Skipping row.", row_idx, doc_id)
                continue

            # Possibly sample these multi-hop chunks
            chosen_multi_hops = sample_multihop_if_needed(multi_hop_chunks)

            if not chosen_multi_hops:
                logger.debug("Row idx={} doc_id={} had multi-hop chunks but none after sampling.", row_idx, doc_id)
                continue

            additional_instructions = stage_cfg.get("additional_instructions", "undergraduate")

            # For each multi-hop chunk group
            for mh_idx, mh in enumerate(chosen_multi_hops):
                if not isinstance(mh, dict):
                    continue

                subchunk_ids = mh.get("chunk_ids", [])
                subchunk_texts = mh.get("chunks_text", [])
                if not subchunk_texts:
                    logger.debug("Empty multi-hop chunk at row_idx={}, doc_id={}. Skipping.", row_idx, doc_id)
                    continue

                # Build the user prompt by enumerating each text_chunk_i
                text_chunks_aggregated = ""
                for i, sc_text in enumerate(subchunk_texts):
                    text_chunks_aggregated += f"<text_chunk_{i}>{sc_text}</text_chunk_{i}>\n"

                user_prompt = MULTI_HOP_QUESTION_GENERATION_USER_PROMPT.format(
                    title=title,
                    document_summary=doc_summary,
                    chunks=text_chunks_aggregated,
                    additional_instructions=additional_instructions
                )
                user_msg = {"role": "user", "content": user_prompt}
                inference_call = InferenceCall(
                    messages=[system_msg, user_msg],
                    tags=["multi_hop_qa"]
                )
                all_inference_calls.append(inference_call)
                # call_index_map: store (row_idx, doc_id, subchunk_ids)
                call_index_map.append((row_idx, doc_id, subchunk_ids))

        if not all_inference_calls:
            logger.warning("No multi-hop inference calls were created. Exiting.")
            return

        logger.info("Sending {} multi-hop calls to inference...", len(all_inference_calls))
        responses_dict = run_inference(
            config=config,
            step_name="multi_hop_question_generation",
            inference_calls=all_inference_calls,
        )

        # We'll store final question rows in memory
        question_dataset_rows: List[Dict[str, Any]] = []

        # For each model that responded, parse the JSON
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
                    question_answer_pairs = json.loads(json_str)
                except Exception as e:
                    logger.warning(
                        "Failed to parse JSON for row={}, doc_id={} (model={}): {}",
                        row_idx, doc_id, model_name, e
                    )
                    continue

                if not isinstance(question_answer_pairs, list):
                    logger.warning(
                        "Expected a list of question-answer pairs; got something else. row={}, doc_id={}, model={}",
                        row_idx, doc_id, model_name
                    )
                    continue

                # For each Q-A pair
                for qap in question_answer_pairs:
                    try:
                        question_text = qap.get("question", "").strip()
                        if not question_text:
                            logger.debug("Empty question for row={}, doc_id={}, skipping pair", row_idx, doc_id)
                            continue

                        self_answer = qap.get("answer", "").strip()

                        difficulty_raw = qap.get("estimated_difficulty", 5)
                        try:
                            difficulty_val = int(difficulty_raw)
                        except (ValueError, TypeError):
                            logger.warning(
                                "Invalid difficulty value '{}' for doc_id={}, defaulting to 5",
                                difficulty_raw,
                                doc_id
                            )
                            difficulty_val = 5

                        qtype = qap.get("question_type", "unknown")
                        if not isinstance(qtype, str):
                            qtype = str(qtype)

                        thought_process = qap.get("thought_process", "")
                        if not isinstance(thought_process, str):
                            thought_process = str(thought_process)

                        cits = qap.get("citations", [])
                        if not isinstance(cits, list):
                            logger.warning(
                                "Citations is not a list for doc_id={}, converting to empty list",
                                doc_id
                            )
                            cits = []

                        row_obj = MultiHopQuestionRow(
                            document_id=doc_id,
                            source_chunk_ids=source_chunk_ids,
                            question=question_text,
                            self_answer=self_answer,
                            estimated_difficulty=difficulty_val,
                            self_assessed_question_type=qtype,
                            generating_model=model_name,
                            thought_process=thought_process,
                            citations=cits
                        )
                        question_dataset_rows.append(row_obj.__dict__)

                    except Exception as e:
                        logger.warning(
                            "Error processing QA pair for doc_id={}, skipping pair: {}",
                            doc_id, e
                        )
                        continue

        if not question_dataset_rows:
            logger.warning("No valid multi-hop question rows produced.")
            return

        logger.info("Constructing multi-hop question dataset with {} rows...", len(question_dataset_rows))

        # Convert to HF Dataset
        try:
            # The first row's keys define the columns
            col_keys = list(question_dataset_rows[0].keys())
            dataset_dict = {
                k: [row[k] for row in question_dataset_rows] for k in col_keys
            }
            question_dataset = Dataset.from_dict(dataset_dict)
        except Exception as e:
            logger.error("Failed to create dataset from multi-hop question rows: {}", e)
            return

        # Save final dataset
        logger.info("Saving multi-hop question dataset as '{}'.", output_dataset_name)
        save_dataset(
            dataset=question_dataset,
            step_name="multi_hop_question_generation",
            config=config,
            output_dataset_name=output_dataset_name,
            output_subset=output_subset
        )
        logger.success("Multi-hop question generation completed successfully.")

    except Exception as e:
        logger.error("Error in multi_hop_question_generation run function: {}", str(e))
        logger.warning("Multi-hop question generation completed with errors - no dataset created.")


# === Helper Functions ===

def _extract_tag_content(text: str, tag: str) -> str:
    """
    Extract content from <tag>...</tag> within the text. Returns empty if not found.
    """
    pattern = fr"<{tag}\s*>([\s\S]*?)</{tag}>"
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""


def _extract_output_json(raw_response: str) -> str:
    """
    Attempt to extract JSON from <output_json> blocks, fenced code with ```json, or
    fallback bracket parsing. Return a string of JSON or empty if none found.
    """
    try:
        # 1. <output_json> block
        extracted = _extract_tag_content(raw_response, "output_json")
        if extracted.strip():
            sanitized = _maybe_strip_triple_backticks(extracted)
            if sanitized.strip():
                return sanitized

        # 2. ```json fenced block
        fence_pattern = r"```json\s*([\s\S]*?)\s*```"
        fm = re.search(fence_pattern, raw_response)
        if fm:
            return fm.group(1).strip()

        # 3. fallback bracket extraction
        cands = _best_effort_json_extract(raw_response)
        if cands:
            return cands[0]

        return ""
    except Exception as e:
        logger.warning("Error in _extract_output_json: {}", e)
        return ""


def _maybe_strip_triple_backticks(text_in: str) -> str:
    """
    Removes triple backticks if the entire text is wrapped in them.
    """
    pattern = r"^\s*```(?:json)?\s*([\s\S]*?)\s*```$"
    mm = re.match(pattern, text_in)
    if mm:
        return mm.group(1)
    return text_in


def _best_effort_json_extract(full_text: str) -> List[str]:
    """
    Look for bracket-delimited content that might be valid JSON.
    Returns a list of candidate strings.
    """
    pattern = r"([\[{].*?[\]}])"
    matches = re.findall(pattern, full_text, flags=re.DOTALL)
    cands = []
    for m in matches:
        if (m.startswith("[") and m.endswith("]")) or (m.startswith("{") and m.endswith("}")):
            cands.append(m.strip())
    return cands
