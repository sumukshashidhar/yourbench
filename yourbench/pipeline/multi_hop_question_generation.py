# yourbench/pipeline/multi_hop_question_generation.py

"""
Multi-Hop Question Generation Module

Minimal approach:
----------------
- Each row in the dataset has "multihop_chunks", which is a list of dicts:
    { "chunk_ids": [...], "chunks_text": [...] }
- For each such multi-hop item, we generate questions that may require
  combining multiple single-hop chunks. We do NOT assign a new multi-hop ID.
- In the final question dataset, we store a list of source_chunk_ids that
  show which single-hop chunks were used. We do not store chunk_uuid or any
  location indices. The final row is basically:
  [
    {
      "document_id": ...,
      "source_chunk_ids": [...],
      "question": ...,
      "self_answer": ...,
      ...
    },
    ...
  ]
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
    Minimal structure for multi-hop question rows:
      - document_id: which doc
      - source_chunk_ids: which single-hop chunks are used
      - question, self_answer, etc.
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
    stage_cfg = config.get("pipeline", {}).get("multi_hop_question_generation", {})
    if not stage_cfg.get("run", False):
        logger.info("multi_hop_question_generation stage is disabled. Skipping.")
        return

    source_dataset_name = stage_cfg["source_dataset_name"]
    output_dataset_name = stage_cfg["output_dataset_name"]
    logger.info("Loading chunked dataset: {}", source_dataset_name)
    dataset = smart_load_dataset(source_dataset_name, config)
    logger.info("Loaded dataset with {} rows.", len(dataset))

    system_msg = {"role": "system", "content": MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT}
    all_inference_calls: List[InferenceCall] = []
    call_index_map: List[tuple] = []

    for row_idx, row in enumerate(dataset):
        doc_summary = row.get("document_summary", "No summary provided.")
        title = row.get("document_filename", f"Document_{row_idx}")
        doc_id = row.get("document_id", f"doc_{row_idx}")

        multi_hop_chunks = row.get("multihop_chunks", [])
        if not isinstance(multi_hop_chunks, list) or not multi_hop_chunks:
            continue

        for mh_idx, mh in enumerate(multi_hop_chunks):
            if not isinstance(mh, dict):
                continue

            subchunk_ids = mh.get("chunk_ids", [])
            subchunk_texts = mh.get("chunks_text", [])
            if not subchunk_texts:
                continue

            # Build multiple <text_chunk_i> tags
            text_chunks_aggregated = ""
            for i, sc_text in enumerate(subchunk_texts):
                text_chunks_aggregated += f"<text_chunk_{i}>{sc_text}</text_chunk_{i}>\n"

            test_audience = stage_cfg.get("test_audience", "undergraduate")
            user_prompt = MULTI_HOP_QUESTION_GENERATION_USER_PROMPT.format(
                title=title,
                document_summary=doc_summary,
                chunks=text_chunks_aggregated,
                test_audience=test_audience
            )
            user_msg = {"role": "user", "content": user_prompt}
            inference_call = InferenceCall(
                messages=[system_msg, user_msg],
                tags=["multi_hop_qa"]
            )
            all_inference_calls.append(inference_call)
            call_index_map.append((row_idx, doc_id, subchunk_ids))

    if not all_inference_calls:
        logger.warning("No multi-hop chunks found. Exiting multi-hop question generation.")
        return

    logger.info("Sending {} calls to inference for multi-hop question generation.", len(all_inference_calls))
    responses_dict = run_inference(
        config=config,
        step_name="multi_hop_question_generation",
        inference_calls=all_inference_calls,
    )

    question_dataset_rows: List[Dict[str, Any]] = []

    for model_name, model_responses in responses_dict.items():
        logger.info("Processing {} responses for model: {}", len(model_responses), model_name)
        if len(model_responses) != len(call_index_map):
            logger.error(
                "Model '{}' returned {} responses, expected {}. Possibly mismatch or truncation.",
                model_name, len(model_responses), len(call_index_map)
            )

        for idx, raw_resp in enumerate(model_responses):
            if idx >= len(call_index_map):
                break
            row_idx, doc_id, source_chunk_ids = call_index_map[idx]

            json_str = _extract_output_json(raw_resp)
            if not json_str.strip():
                logger.warning("No parseable JSON for row={}, doc_id={} (model={}).", row_idx, doc_id, model_name)
                continue

            try:
                question_answer_pairs = json.loads(json_str)
            except Exception as e:
                logger.warning("Failed to parse JSON for row={}, doc_id={} (model={}): {}", row_idx, doc_id, model_name, e)
                continue

            if not isinstance(question_answer_pairs, list):
                logger.warning("JSON is not a list for row={}, doc_id={} (model={}).", row_idx, doc_id, model_name)
                continue

            for qap in question_answer_pairs:
                question = qap.get("question", "")
                self_answer = qap.get("answer", "")
                difficulty = qap.get("estimated_difficulty", 5)
                qtype = qap.get("question_type", "unknown")
                thought_process = qap.get("thought_process", "")
                cits = qap.get("citations", [])

                row_obj = MultiHopQuestionRow(
                    document_id=doc_id,
                    source_chunk_ids=source_chunk_ids,
                    question=question,
                    self_answer=self_answer,
                    estimated_difficulty=difficulty,
                    self_assessed_question_type=qtype,
                    generating_model=model_name,
                    thought_process=thought_process,
                    citations=cits
                )
                question_dataset_rows.append(row_obj.__dict__)

    if not question_dataset_rows:
        logger.warning("No valid question rows produced from multi-hop generation.")
        return

    logger.info("Constructing multi-hop question-level dataset with {} rows...", len(question_dataset_rows))
    question_dataset = Dataset.from_dict({
        k: [d[k] for d in question_dataset_rows]
        for k in question_dataset_rows[0].keys()
    })

    logger.info("Saving multi-hop question dataset as '{}'.", output_dataset_name)
    save_dataset(
        dataset=question_dataset,
        step_name="multi_hop_question_generation",
        config=config,
        output_dataset_name=output_dataset_name
    )
    logger.success("Multi-hop question generation completed successfully.")


def _extract_tag_content(text: str, tag: str) -> str:
    pattern = fr"<{tag}\s*>([\s\S]*?)</{tag}>"
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""


def _extract_output_json(raw_response: str) -> str:
    # 1. Check <output_json> block
    extracted = _extract_tag_content(raw_response, "output_json")
    if extracted.strip():
        sanitized = _maybe_strip_triple_backticks(extracted)
        if sanitized.strip():
            return sanitized

    # 2. Check ```json fenced block
    fence_pattern = r"```json\s*([\s\S]*?)\s*```"
    fm = re.search(fence_pattern, raw_response)
    if fm:
        return fm.group(1).strip()

    # 3. fallback bracket extraction
    cands = _best_effort_json_extract(raw_response)
    return cands[0] if cands else ""


def _maybe_strip_triple_backticks(text_in: str) -> str:
    pattern = r"^\s*```(?:json)?\s*([\s\S]*?)\s*```$"
    mm = re.match(pattern, text_in)
    if mm:
        return mm.group(1)
    return text_in


def _best_effort_json_extract(full_text: str) -> List[str]:
    pattern = r"([\[{].*?[\]}])"
    matches = re.findall(pattern, full_text, flags=re.DOTALL)
    cands = []
    for m in matches:
        if (m.startswith("[") and m.endswith("]")) or (m.startswith("{") and m.endswith("}")):
            cands.append(m.strip())
    return cands
