from typing import Any

import tiktoken
from loguru import logger

from datasets import Dataset
from yourbench.utils.prompts import (
    SUMMARIZATION_USER_PROMPT,
    COMBINE_SUMMARIES_USER_PROMPT,
)
from yourbench.utils.chunking_utils import split_into_token_chunks
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.parsing_engine import extract_content_from_xml_tags
from yourbench.utils.inference_engine import InferenceCall, run_inference


############################
# Internal helper functions #
############################


def _build_chunk_calls(
    dataset: Dataset,
    max_tokens: int,
    overlap: int,
    encoding_name: str,
) -> tuple[list[InferenceCall], list[tuple[int, int]]]:
    """Prepare inference calls for first-level chunk summaries.

    Returns:
        A tuple containing:
        - A list of inference calls.
        - A list of mappings, where each mapping is a tuple (doc_idx, chunk_idx)
          aligning each call to its document and chunk index. chunk_idx is -1 for
          documents treated as a single chunk.
    """
    calls: list[InferenceCall] = []
    mapping: list[tuple[int, int]] = []  # (doc_index, chunk_index)

    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        error_message = str(e)
        truncated_error = error_message[:60] + ("…" if len(error_message) > 60 else "")
        logger.warning(
            f"Unknown / unavailable encoding '{encoding_name}'. Falling back to 'cl100k_base' ({truncated_error})"
        )
        enc = tiktoken.get_encoding("cl100k_base")

    for doc_idx, doc_text in enumerate(dataset["document_text"]):
        token_len = len(enc.encode(doc_text))
        if token_len <= max_tokens:  # treat as single chunk (chunk_idx = -1)
            prompt = SUMMARIZATION_USER_PROMPT.format(document=doc_text)
            calls.append(InferenceCall(messages=[{"role": "user", "content": prompt}], tags=["chunk_summary"]))
            mapping.append((doc_idx, -1))
            continue

        chunks = split_into_token_chunks(
            doc_text,
            chunk_tokens=max_tokens,
            overlap=overlap,
            encoding_name=encoding_name,
        )
        for chunk_idx, chunk in enumerate(chunks):
            prompt = SUMMARIZATION_USER_PROMPT.format(document=chunk)
            calls.append(InferenceCall(messages=[{"role": "user", "content": prompt}], tags=["chunk_summary"]))
            mapping.append((doc_idx, chunk_idx))

    logger.info(f"Prepared {len(calls)} chunk-level inference calls.")
    return calls, mapping


def _collect_chunk_summaries(
    response_dict: dict[str, list[str]],
    mapping: list[tuple[int, int]],
    num_docs: int,
) -> tuple[str, list[list[str]], list[list[str]]]:
    """Re-orders raw model responses back into per-document lists of summaries."""
    if not response_dict:
        return "", [], []

    model_name = list(response_dict.keys())[0]
    responses = response_dict[model_name]

    if len(responses) != len(mapping):
        logger.warning(f"Response count {len(responses)} ≠ mapping count {len(mapping)} – truncating/min-padding.")
        diff = len(mapping) - len(responses)
        if diff > 0:
            responses.extend([""] * diff)
        else:
            responses = responses[: len(mapping)]

    raw_by_doc: list[list[str]] = [[] for _ in range(num_docs)]
    cleaned_by_doc: list[list[str]] = [[] for _ in range(num_docs)]

    for resp, (doc_idx, _chunk_idx) in zip(responses, mapping):
        raw_by_doc[doc_idx].append(resp)
        summary_content = extract_content_from_xml_tags(resp, "chunk_summary") or extract_content_from_xml_tags(
            resp, "final_summary"
        )
        cleaned_by_doc[doc_idx].append(summary_content.strip() if summary_content else "")

    return model_name, raw_by_doc, cleaned_by_doc


def _build_combine_calls(summaries_by_doc: list[list[str]]) -> tuple[list[InferenceCall], list[int]]:
    """Prepare second-stage calls to merge multiple chunk summaries into a single summary."""
    calls: list[InferenceCall] = []
    doc_indices_for_combine: list[int] = []
    skipped_doc_count = 0

    for doc_idx, chunk_summaries in enumerate(summaries_by_doc):
        if len(chunk_summaries) <= 1:  # Already a single summary (or empty), skip combine
            skipped_doc_count += 1
            continue

        valid_summaries = [s for s in chunk_summaries if s]
        if not valid_summaries:
            skipped_doc_count += 1
            continue

        bullet_list = "\\n".join(f"- {s}" for s in valid_summaries)
        prompt = COMBINE_SUMMARIES_USER_PROMPT.format(chunk_summaries=bullet_list)
        calls.append(InferenceCall(messages=[{"role": "user", "content": prompt}], tags=["merge_summary"]))
        doc_indices_for_combine.append(doc_idx)

    logger.info(
        f"Prepared {len(calls)} combine-stage inference calls ({skipped_doc_count} docs skipped – single/empty chunk list)."
    )
    return calls, doc_indices_for_combine


def _merge_final_summaries(
    current_final_summaries: list[str],
    combined_responses: list[str],
    doc_indices_to_update: list[int],
) -> list[str]:
    """Integrates combined summaries into the list of final summaries."""
    updated_final_summaries = current_final_summaries.copy()

    for resp, doc_idx in zip(combined_responses, doc_indices_to_update):
        parsed_summary = extract_content_from_xml_tags(resp, "final_summary")
        updated_final_summaries[doc_idx] = parsed_summary.strip() if parsed_summary else "No summary available."
    return updated_final_summaries


#################
# Stage runner  #
#################


def run(config: dict[str, Any]) -> None:
    """Executes the hierarchical summarization pipeline."""
    stage_cfg = config.get("pipeline", {}).get("summarization", {})
    if not stage_cfg.get("run", False):
        logger.info("Summarization stage disabled – skipping.")
        return

    max_tokens: int = stage_cfg.get("max_tokens", 16384)
    overlap: int = stage_cfg.get("token_overlap", 128)
    encoding_name: str = stage_cfg.get("encoding_name", "cl100k_base")

    logger.info("=== Summarization v2 – map-reduce ===")

    dataset = custom_load_dataset(config=config, subset="ingested")
    if not dataset or len(dataset) == 0:
        logger.warning("Ingested dataset is empty or None – nothing to summarise.")
        return
    logger.info(f"Loaded {len(dataset)} documents for summarisation.")

    chunk_calls, call_map = _build_chunk_calls(dataset, max_tokens, overlap, encoding_name)
    chunk_responses_dict = run_inference(config=config, step_name="summarization_chunk", inference_calls=chunk_calls)
    model_name, raw_chunk_summaries_by_doc, cleaned_chunk_summaries_by_doc = _collect_chunk_summaries(
        chunk_responses_dict, call_map, len(dataset)
    )

    combine_calls, doc_indices_for_combine = _build_combine_calls(cleaned_chunk_summaries_by_doc)

    raw_combined_summaries: list[str] = []
    if combine_calls:
        combine_responses_dict = run_inference(
            config=config, step_name="summarization_combine", inference_calls=combine_calls
        )
        if combine_responses_dict:
            combine_model_name = list(combine_responses_dict.keys())[0]
            if combine_model_name != model_name and model_name:
                logger.warning(f"Different model used in combine stage: {combine_model_name} vs {model_name}")
            raw_combined_summaries = combine_responses_dict.get(combine_model_name, [])
        else:
            raw_combined_summaries = [""] * len(doc_indices_for_combine)

    final_document_summaries: list[str] = [
        summaries[0] if summaries else "" for summaries in cleaned_chunk_summaries_by_doc
    ]

    if combine_calls and raw_combined_summaries:
        final_document_summaries = _merge_final_summaries(
            final_document_summaries, raw_combined_summaries, doc_indices_for_combine
        )

    full_raw_combined_summaries = [""] * len(dataset)
    for i, doc_idx in enumerate(doc_indices_for_combine):
        if i < len(raw_combined_summaries):
            full_raw_combined_summaries[doc_idx] = raw_combined_summaries[i]

    dataset = dataset.add_column("raw_chunk_summaries", raw_chunk_summaries_by_doc)
    dataset = dataset.add_column("chunk_summaries", cleaned_chunk_summaries_by_doc)
    dataset = dataset.add_column("raw_document_summary", full_raw_combined_summaries)
    dataset = dataset.add_column("document_summary", final_document_summaries)
    effective_model_name = model_name if model_name else "unknown"
    dataset = dataset.add_column("summarization_model", [effective_model_name] * len(dataset))

    custom_save_dataset(dataset=dataset, config=config, subset="summarized")
    logger.success(f"Hierarchical summarisation completed ({len(dataset)} documents).")
