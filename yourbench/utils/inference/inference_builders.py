from typing import List
from dataclasses import dataclass

from loguru import logger

from yourbench.utils.prompts import QUESTION_GENERATION_USER_PROMPT, MULTI_HOP_QUESTION_GENERATION_USER_PROMPT
from yourbench.utils.chunking_utils import sample_multihop_groups, sample_single_hop_chunks
from yourbench.utils.inference.inference_core import InferenceCall


@dataclass
class InferenceJob:
    inference_calls: List[InferenceCall]


def build_single_shot_inference_calls(dataset, system_msg, stage_cfg, sampling_cfg):
    calls = []
    index_map = []

    for idx, row in enumerate(dataset):
        document_chunks = row.get("chunks") or []
        selected_chunks = sample_single_hop_chunks(document_chunks, sampling_cfg)

        for ch_idx, chunk in enumerate(selected_chunks):
            chunk_id = chunk.get("chunk_id", f"{idx}_{ch_idx}")
            chunk_text = chunk.get("chunk_text", "")
            user_msg = {
                "role": "user",
                "content": QUESTION_GENERATION_USER_PROMPT.format(
                    title=row.get("document_filename", f"doc_{idx}"),
                    document_summary=row.get("document_summary", ""),
                    text_chunk=chunk_text,
                    additional_instructions=stage_cfg.get("additional_instructions", ""),
                ),
            }
            calls.append(InferenceCall(messages=[system_msg, user_msg], tags=["single_shot_qa"]))
            index_map.append((idx, row.get("document_id", f"doc_{idx}"), chunk_id))

    return calls, index_map


def build_multi_hop_inference_calls(dataset, system_msg, stage_cfg):
    calls = []
    index_map = []

    for idx, row in enumerate(dataset):
        groups = sample_multihop_groups(row.get("multihop_chunks") or [], stage_cfg.get("chunk_sampling", {}))
        for group in groups:
            # TODO how it's possible here?
            if not isinstance(group, dict):
                logger.warning("Multihop groups are not a dict, skipping")
                continue
            chunk_ids = group.get("chunk_ids", [])
            texts = group.get("chunks_text", [])
            if not texts:
                logger.warning("Chunks texts are empty, skipping")
                continue
            full_text = "".join([f"<text_chunk_{i}>{t}</text_chunk_{i}>\n" for i, t in enumerate(texts)])
            user_msg = {
                "role": "user",
                "content": MULTI_HOP_QUESTION_GENERATION_USER_PROMPT.format(
                    title=row.get("document_filename", ""),
                    document_summary=row.get("document_summary", ""),
                    chunks=full_text,
                    additional_instructions=stage_cfg.get("additional_instructions", ""),
                ),
            }
            calls.append(InferenceCall(messages=[system_msg, user_msg], tags=["multi_hop_qa"]))
            index_map.append((idx, row.get("document_id", f"doc_{idx}"), chunk_ids))

    return calls, index_map
