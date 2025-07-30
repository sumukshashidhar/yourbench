import time
from typing import Any, Dict, List
from dataclasses import dataclass

from loguru import logger

# User prompts are now passed via configuration
from yourbench.utils.chunking_utils import sample_multihop_groups, sample_single_hop_chunks
from yourbench.utils.inference.inference_core import InferenceCall


@dataclass
class InferenceJob:
    """Enhanced inference job with metadata tracking."""

    inference_calls: List[InferenceCall]
    job_metadata: Dict[str, Any] = None
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.job_metadata is None:
            self.job_metadata = {}


@dataclass
class BuilderMetrics:
    """Metrics for tracking inference call generation."""

    total_documents: int = 0
    total_chunks_processed: int = 0
    total_calls_generated: int = 0
    skipped_chunks: int = 0
    avg_chunk_length: float = 0.0
    processing_time: float = 0.0
    error_count: int = 0
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def _calculate_chunk_stats(chunks: List[Dict]) -> Dict[str, float]:
    """Calculate statistics for chunks."""
    if not chunks:
        return {"avg_length": 0.0, "total_length": 0, "count": 0}

    lengths = [len(chunk.get("chunk_text", "")) for chunk in chunks]
    return {
        "avg_length": sum(lengths) / len(lengths),
        "total_length": sum(lengths),
        "count": len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


def build_single_shot_inference_calls(dataset, system_msg, stage_cfg, sampling_cfg):
    """Build single-shot inference calls with enhanced tracking."""
    start_time = time.time()
    calls = []
    index_map = []
    metrics = BuilderMetrics()

    logger.info(f"Building single-shot inference calls for {len(dataset)} documents")

    for idx, row in enumerate(dataset):
        try:
            metrics.total_documents += 1
            document_chunks = row.get("chunks") or []

            if not document_chunks:
                metrics.warnings.append(f"Document {idx} has no chunks")
                continue

            selected_chunks = sample_single_hop_chunks(document_chunks, sampling_cfg)
            chunk_stats = _calculate_chunk_stats(selected_chunks)

            for ch_idx, chunk in enumerate(selected_chunks):
                try:
                    metrics.total_chunks_processed += 1
                    chunk_id = chunk.get("chunk_id", f"{idx}_{ch_idx}")
                    chunk_text = chunk.get("chunk_text", "")

                    if not chunk_text.strip():
                        metrics.skipped_chunks += 1
                        metrics.warnings.append(f"Empty chunk {chunk_id}")
                        continue

                    # Get additional instructions with fallback
                    additional_instructions = ""
                    if hasattr(stage_cfg, "additional_instructions"):
                        additional_instructions = stage_cfg.additional_instructions
                    elif isinstance(stage_cfg, dict):
                        additional_instructions = stage_cfg.get("additional_instructions", "")

                    user_msg = {
                        "role": "user",
                        "content": stage_cfg.single_shot_user_prompt.format(
                            title=row.get("document_filename", f"doc_{idx}"),
                            document_summary=row.get("document_summary", ""),
                            text_chunk=chunk_text,
                            additional_instructions=additional_instructions,
                        ),
                    }

                    # Enhanced tags with metadata
                    tags = [
                        "single_shot_qa",
                        f"doc_{idx}",
                        f"chunk_{ch_idx}",
                        f"chunk_len_{len(chunk_text)}",
                    ]

                    # Add document type if available
                    if "document_type" in row:
                        tags.append(f"type_{row['document_type']}")

                    call = InferenceCall(
                        messages=[system_msg, user_msg],
                        tags=tags,
                        temperature=getattr(stage_cfg, "temperature", None),
                        max_retries=getattr(stage_cfg, "max_retries", 12),
                    )

                    calls.append(call)
                    index_map.append((idx, row.get("document_id", f"doc_{idx}"), chunk_id))
                    metrics.total_calls_generated += 1

                except Exception as e:
                    metrics.error_count += 1
                    metrics.warnings.append(f"Error processing chunk {ch_idx} in document {idx}: {str(e)}")
                    logger.warning(f"Error processing chunk {ch_idx} in document {idx}: {e}")
                    continue

            # Log chunk statistics for this document
            if chunk_stats["count"] > 0:
                logger.debug(
                    f"Document {idx}: processed {chunk_stats['count']} chunks, "
                    f"avg_length={chunk_stats['avg_length']:.0f}, "
                    f"total_length={chunk_stats['total_length']}"
                )

        except Exception as e:
            metrics.error_count += 1
            metrics.warnings.append(f"Error processing document {idx}: {str(e)}")
            logger.error(f"Error processing document {idx}: {e}")
            continue

    metrics.processing_time = time.time() - start_time

    # Calculate average chunk length
    if metrics.total_chunks_processed > 0:
        total_length = sum(len(call.messages[-1]["content"]) for call in calls)
        metrics.avg_chunk_length = total_length / metrics.total_chunks_processed

    # Log final metrics
    logger.info(
        f"Single-shot builder completed: {metrics.total_calls_generated} calls from "
        f"{metrics.total_documents} documents, {metrics.total_chunks_processed} chunks processed "
        f"(skipped: {metrics.skipped_chunks}, errors: {metrics.error_count}) "
        f"in {metrics.processing_time:.2f}s"
    )

    if metrics.warnings:
        logger.warning(f"Builder warnings: {len(metrics.warnings)} total")
        for warning in metrics.warnings[:5]:  # Show first 5 warnings
            logger.warning(f"  - {warning}")
        if len(metrics.warnings) > 5:
            logger.warning(f"  ... and {len(metrics.warnings) - 5} more warnings")

    return calls, index_map


def build_multi_hop_inference_calls(dataset, system_msg, stage_cfg):
    """Build multi-hop inference calls with enhanced tracking."""
    start_time = time.time()
    calls = []
    index_map = []
    metrics = BuilderMetrics()

    logger.info(f"Building multi-hop inference calls for {len(dataset)} documents")

    for idx, row in enumerate(dataset):
        try:
            metrics.total_documents += 1
            multihop_chunks = row.get("multihop_chunks") or []

            if not multihop_chunks:
                metrics.warnings.append(f"Document {idx} has no multihop chunks")
                continue

            # Get chunk sampling configuration
            chunk_sampling = {}
            if hasattr(stage_cfg, "chunk_sampling"):
                chunk_sampling = stage_cfg.chunk_sampling
            elif isinstance(stage_cfg, dict):
                chunk_sampling = stage_cfg.get("chunk_sampling", {})

            groups = sample_multihop_groups(multihop_chunks, chunk_sampling)

            for group_idx, group in enumerate(groups):
                try:
                    if not isinstance(group, dict):
                        metrics.warnings.append(f"Multihop group {group_idx} in document {idx} is not a dict")
                        logger.warning(f"Multihop group {group_idx} in document {idx} is not a dict, skipping")
                        continue

                    chunk_ids = group.get("chunk_ids", [])
                    texts = group.get("chunks_text", [])

                    if not texts:
                        metrics.warnings.append(f"Group {group_idx} in document {idx} has empty chunks_text")
                        logger.warning(f"Group {group_idx} in document {idx} has empty chunks_text, skipping")
                        continue

                    metrics.total_chunks_processed += len(texts)

                    # Format chunks with XML-like tags
                    full_text = "".join([f"<text_chunk_{i}>{t}</text_chunk_{i}>\n" for i, t in enumerate(texts)])

                    # Get additional instructions with fallback
                    additional_instructions = ""
                    if hasattr(stage_cfg, "additional_instructions"):
                        additional_instructions = stage_cfg.additional_instructions
                    elif isinstance(stage_cfg, dict):
                        additional_instructions = stage_cfg.get("additional_instructions", "")

                    user_msg = {
                        "role": "user",
                        "content": stage_cfg.multi_hop_user_prompt.format(
                            title=row.get("document_filename", f"doc_{idx}"),
                            document_summary=row.get("document_summary", ""),
                            chunks=full_text,
                            additional_instructions=additional_instructions,
                        ),
                    }

                    # Enhanced tags with metadata
                    tags = [
                        "multi_hop_qa",
                        f"doc_{idx}",
                        f"group_{group_idx}",
                        f"chunks_{len(texts)}",
                        f"total_len_{len(full_text)}",
                    ]

                    # Add document type if available
                    if "document_type" in row:
                        tags.append(f"type_{row['document_type']}")

                    # Add chunk count category
                    if len(texts) <= 2:
                        tags.append("few_chunks")
                    elif len(texts) <= 5:
                        tags.append("medium_chunks")
                    else:
                        tags.append("many_chunks")

                    call = InferenceCall(
                        messages=[system_msg, user_msg],
                        tags=tags,
                        temperature=getattr(stage_cfg, "temperature", None),
                        max_retries=getattr(stage_cfg, "max_retries", 12),
                    )

                    calls.append(call)
                    index_map.append((idx, row.get("document_id", f"doc_{idx}"), chunk_ids))
                    metrics.total_calls_generated += 1

                    # Log group statistics
                    avg_chunk_length = sum(len(t) for t in texts) / len(texts)
                    logger.debug(
                        f"Document {idx} group {group_idx}: {len(texts)} chunks, "
                        f"avg_length={avg_chunk_length:.0f}, total_length={len(full_text)}"
                    )

                except Exception as e:
                    metrics.error_count += 1
                    metrics.warnings.append(f"Error processing group {group_idx} in document {idx}: {str(e)}")
                    logger.warning(f"Error processing group {group_idx} in document {idx}: {e}")
                    continue

        except Exception as e:
            metrics.error_count += 1
            metrics.warnings.append(f"Error processing document {idx}: {str(e)}")
            logger.error(f"Error processing document {idx}: {e}")
            continue

    metrics.processing_time = time.time() - start_time

    # Calculate average chunk length
    if metrics.total_chunks_processed > 0:
        total_length = sum(len(call.messages[-1]["content"]) for call in calls)
        metrics.avg_chunk_length = total_length / metrics.total_chunks_processed

    # Log final metrics
    logger.info(
        f"Multi-hop builder completed: {metrics.total_calls_generated} calls from "
        f"{metrics.total_documents} documents, {metrics.total_chunks_processed} chunks processed "
        f"(errors: {metrics.error_count}) in {metrics.processing_time:.2f}s"
    )

    if metrics.warnings:
        logger.warning(f"Builder warnings: {len(metrics.warnings)} total")
        for warning in metrics.warnings[:5]:  # Show first 5 warnings
            logger.warning(f"  - {warning}")
        if len(metrics.warnings) > 5:
            logger.warning(f"  ... and {len(metrics.warnings) - 5} more warnings")

    return calls, index_map


def get_builder_performance_summary(calls: List[InferenceCall], processing_time: float) -> Dict[str, Any]:
    """Generate performance summary for builder operations."""
    if not calls:
        return {"total_calls": 0, "processing_time": processing_time}

    # Analyze tags to understand call distribution
    tag_counts = {}
    message_lengths = []

    for call in calls:
        for tag in call.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Get total message length
        total_length = sum(len(str(msg.get("content", ""))) for msg in call.messages)
        message_lengths.append(total_length)

    avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0

    return {
        "total_calls": len(calls),
        "processing_time": processing_time,
        "avg_message_length": avg_message_length,
        "min_message_length": min(message_lengths) if message_lengths else 0,
        "max_message_length": max(message_lengths) if message_lengths else 0,
        "tag_distribution": tag_counts,
        "calls_per_second": len(calls) / processing_time if processing_time > 0 else 0,
    }
