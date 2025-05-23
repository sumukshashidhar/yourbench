# summarization.py
# =============================================================================
# Author: @sumukshashidhar
#
# Module: Summarization Pipeline Stage
# =============================================================================
"""
Summarization Stage
===================

This module handles the summarization stage of the YourBench pipeline. It takes
documents (with their raw text) and generates concise yet comprehensive summaries
for each document.

Usage:
------
1. Ensure the pipeline configuration has an entry for the `summarization` stage
   with the desired settings. For example:

   summarization:
     run: true
     timeout_seconds: 300

2. When the pipeline runs, it loads the target dataset, calls the summarization
   model(s) to produce summaries, logs intermediate steps, and saves the updated
   dataset with new columns:
     - raw_document_summary
     - document_summary
     - summarization_model

Error Handling & Logging:
-------------------------
- All errors are logged using `loguru` to `logs/summarization.log`.
- The stage attempts to proceed with partial data even if some calls fail, never
  abruptly terminating the pipeline.

Important Notes:
----------------
- This stage relies on the `run_inference` utility function from yourbench.utils.inference_engine
  for concurrency, timeouts, and model management.
- Summaries are extracted from the model's output by parsing <final_summary> XML tags.
- If no valid summary is found, the pipeline substitutes a fallback string.

See Also:
---------
- yourbench.utils.inference_engine for concurrency logic
- yourbench.utils.dataset_engine for loading/saving dataset
"""

from __future__ import annotations
from typing import Any, List, Tuple

import tiktoken
from loguru import logger

from datasets import Dataset
from yourbench.utils.prompts import (
    COMBINE_SUMMARIES_USER_PROMPT,
    CHUNK_SUMMARIZATION_USER_PROMPT,
)
from yourbench.utils.chunking_utils import split_into_token_chunks
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.parsing_engine import extract_content_from_xml_tags
from yourbench.utils.inference_engine import InferenceCall, run_inference


def _build_chunk_calls(
    dataset: Dataset,
    max_tokens: int,
    overlap: int,
    encoding_name: str,
) -> Tuple[List[InferenceCall], List[Tuple[int, int]]]:
    """Prepare inference calls for first-level chunk summaries.

    Returns
    -------
    (calls, mapping) where *mapping* aligns each call to (doc_idx, chunk_idx).
    """
    calls: List[InferenceCall] = []
    mapping: List[Tuple[int, int]] = []  # (doc_index, chunk_index)

    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception as e:  # KeyError on unknown name, ValueError on bad cache
        logger.warning(
            "Unknown / unavailable encoding '{}'.  Falling back to 'cl100k_base' ({})",
            encoding_name,
            str(e)[:60] + ("…" if len(str(e)) > 60 else ""),
        )
        enc = tiktoken.get_encoding("cl100k_base")

    for doc_idx, doc_text in enumerate(dataset["document_text"]):
        token_len = len(enc.encode(doc_text, disallowed_special=()))
        if token_len <= max_tokens:  # treat as single chunk (chunk_idx = -1)
            prompt = CHUNK_SUMMARIZATION_USER_PROMPT.format(chunk=doc_text)
            calls.append(InferenceCall(messages=[{"role": "user", "content": prompt}], tags=["chunk_summary"]))
            mapping.append((doc_idx, -1))
            continue

        # Long doc ⇒ split & create a call per chunk
        chunks = split_into_token_chunks(
            doc_text,
            chunk_tokens=max_tokens,
            overlap=overlap,
            encoding_name=encoding_name,
        )
        for chunk_idx, chunk in enumerate(chunks):
            prompt = CHUNK_SUMMARIZATION_USER_PROMPT.format(chunk=chunk)
            calls.append(InferenceCall(messages=[{"role": "user", "content": prompt}], tags=["chunk_summary"]))
            mapping.append((doc_idx, chunk_idx))

    logger.info("Prepared {} chunk-level inference calls.", len(calls))
    return calls, mapping


def _collect_chunk_summaries(
    response_dict: dict[str, List[str]],
    mapping: List[Tuple[int, int]],
    num_docs: int,
) -> Tuple[str, List[List[str]], List[List[str]]]:
    """Re-orders raw model responses back into per-document lists.

    model_name: str is guaranteed to be set, as we return early if response_dict is empty.
    """
    if not response_dict:
        return "", [], []

    model_name = list(response_dict.keys())[0]
    responses = response_dict[model_name]

    # Ensure response count matches call count
    if len(responses) != len(mapping):
        logger.warning("Response count {} ≠ mapping count {} – truncating/min-padding.", len(responses), len(mapping))
        # pad / trim
        diff = len(mapping) - len(responses)
        if diff > 0:
            responses.extend([""] * diff)
        else:
            responses = responses[: len(mapping)]

    # bucket by doc
    raw_by_doc: List[List[str]] = [[] for _ in range(num_docs)]
    cleaned_by_doc: List[List[str]] = [[] for _ in range(num_docs)]

    for resp, (doc_idx, _chunk_idx) in zip(responses, mapping):
        raw_by_doc[doc_idx].append(resp)
        summary = extract_content_from_xml_tags(resp, "chunk_summary") or extract_content_from_xml_tags(
            resp, "final_summary"
        )
        cleaned_by_doc[doc_idx].append(summary.strip() if summary else "")

    return model_name, raw_by_doc, cleaned_by_doc


def _build_combine_calls(summaries_by_doc: List[List[str]]) -> Tuple[List[InferenceCall], List[int]]:
    """Prepare second-stage calls that merge chunk summaries into one summary."""
    calls: List[InferenceCall] = []
    doc_indices: List[int] = []
    skipped = 0  # MOD: track how many docs are trivially short

    for doc_idx, chunk_summaries in enumerate(summaries_by_doc):
        if len(chunk_summaries) <= 1:  # already short ⇒ skip combine
            skipped += 1
            continue
        bullet_list = "\n".join(f"- {s}" for s in chunk_summaries if s)
        prompt = COMBINE_SUMMARIES_USER_PROMPT.format(chunk_summaries=bullet_list)
        calls.append(InferenceCall(messages=[{"role": "user", "content": prompt}], tags=["merge_summary"]))
        doc_indices.append(doc_idx)

    logger.info("Prepared {} reducer calls ({} docs skipped – single / empty chunk).", len(calls), skipped)  # NEW line
    return calls, doc_indices


def _merge_final_summaries(
    existing_singletons: List[str],
    combine_responses: List[str],
    doc_indices: List[int],
) -> List[str]:
    """Blend reducer results with already-final single-chunk docs."""
    final_summaries = existing_singletons.copy()

    for resp, doc_idx in zip(combine_responses, doc_indices):
        parsed = extract_content_from_xml_tags(resp, "final_summary")
        final_summaries[doc_idx] = parsed.strip() if parsed else "No summary available."
    return final_summaries


def run(config: dict[str, Any]) -> None:
    stage_cfg = config.get("pipeline", {}).get("summarization", {})
    if not stage_cfg.get("run", False):
        logger.info("Summarization stage disabled – skipping.")
        return

    max_tokens = stage_cfg.get("max_tokens", 16384)
    overlap = stage_cfg.get("token_overlap", 128)
    encoding_name = stage_cfg.get("encoding_name", "cl100k_base")

    logger.info("=== Summarization v2 – map-reduce ===")

    # 1) Load dataset produced by ingestion
    dataset = custom_load_dataset(config=config, subset="ingested")
    if len(dataset) == 0:
        logger.warning("Ingested dataset empty – nothing to summarise.")
        return
    logger.info("Loaded {} documents for summarisation.", len(dataset))

    # 2) First pass – chunk summaries
    chunk_calls, call_map = _build_chunk_calls(dataset, max_tokens, overlap, encoding_name)
    chunk_resp = run_inference(config=config, step_name="summarization_chunk", inference_calls=chunk_calls)
    model_name, raw_chunk_by_doc, clean_chunk_by_doc = _collect_chunk_summaries(chunk_resp, call_map, len(dataset))

    # 3) Second pass – combine summaries where needed
    combine_calls, doc_indices = _build_combine_calls(clean_chunk_by_doc)

    if len(doc_indices) == 0:
        logger.info("All documents were short enough for single-pass summarization.")

    combine_summaries_raw: List[str] = []
    if combine_calls:
        combine_resp = run_inference(config=config, step_name="summarization_combine", inference_calls=combine_calls)
        combine_model = list(combine_resp.keys())[0] if combine_resp else model_name
        if combine_model != model_name:
            logger.warning("Different model used in reducer stage: {} vs {}", combine_model, model_name)
        combine_summaries_raw = combine_resp.get(combine_model, []) if combine_resp else []

    # produce final list matching dataset order
    # Start with single-chunk docs: take their sole summary
    final_summaries = [docs[0] if docs else "" for docs in clean_chunk_by_doc]
    if combine_calls:
        final_summaries = _merge_final_summaries(final_summaries, combine_summaries_raw, doc_indices)

    # 4) Add columns & persist
    dataset = dataset.add_column("raw_chunk_summaries", raw_chunk_by_doc)
    dataset = dataset.add_column("chunk_summaries", clean_chunk_by_doc)

    # Fill summaries only for combined docs; others stay empty for alignment.
    raw_document_summary = [""] * len(dataset)
    if combine_calls:
        for idx, doc_idx in enumerate(doc_indices):
            raw_document_summary[doc_idx] = combine_summaries_raw[idx]

    dataset = dataset.add_column("raw_document_summary", raw_document_summary)

    dataset = dataset.add_column("document_summary", final_summaries)
    dataset = dataset.add_column("summarization_model", [model_name] * len(dataset))

    custom_save_dataset(dataset=dataset, config=config, subset="summarized")
    logger.success("Hierarchical summarisation completed ({} documents).", len(dataset))
