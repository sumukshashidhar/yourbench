import time
from typing import Any
from dataclasses import asdict, dataclass
from collections.abc import Sequence

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from yourbench.utils.chunking_utils import split_into_token_chunks
from yourbench.utils.dataset_engine import get_hf_settings, custom_load_dataset, custom_save_dataset


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for chunking parameters."""

    max_tokens: int = 256
    h_min: int = 2
    h_max: int = 5
    num_multihops_factor: int = 1


@dataclass(frozen=True)
class SingleHopChunk:
    """A single text chunk with its identifier."""

    chunk_id: str
    chunk_text: str


@dataclass(frozen=True)
class MultiHopChunk:
    """A combination of multiple single-hop chunks."""

    chunk_ids: list[str]
    chunks_text: list[str]


def extract_config(config: dict[str, Any]) -> ChunkingConfig:
    """Extract chunking configuration from pipeline config."""
    chunking_params = config.get("pipeline", {}).get("chunking", {}).get("chunking_configuration", {})
    return ChunkingConfig(
        max_tokens=chunking_params.get("l_max_tokens", 256),
        h_min=chunking_params.get("h_min", 2),
        h_max=chunking_params.get("h_max", 5),
        num_multihops_factor=chunking_params.get("num_multihops_factor", 1),
    )


def chunk_document(text: str, doc_id: str, max_tokens: int) -> list[SingleHopChunk]:
    """
    Chunk a document into segments based on token count.

    Args:
        text: Document text to chunk
        doc_id: Unique document identifier
        max_tokens: Maximum tokens per chunk

    Returns:
        List of single-hop chunks
    """
    if not text or not text.strip():
        return []

    chunk_texts = split_into_token_chunks(text, chunk_tokens=max_tokens, overlap=0)
    return [SingleHopChunk(chunk_id=f"{doc_id}_{i}", chunk_text=chunk) for i, chunk in enumerate(chunk_texts)]


def create_multihop_chunks(
    chunks: Sequence[SingleHopChunk], h_min: int, h_max: int, num_multihops_factor: int
) -> list[MultiHopChunk]:
    """
    Create multi-hop chunks by randomly sampling combinations of single-hop chunks.

    Args:
        chunks: List of single-hop chunks
        h_min: Minimum chunks per multi-hop
        h_max: Maximum chunks per multi-hop
        num_multihops_factor: Factor to determine number of multi-hops

    Returns:
        List of multi-hop chunks
    """
    if not chunks or h_min > len(chunks) or h_min > h_max or h_min <= 0:
        return []

    total_chunks = len(chunks)
    effective_h_max = min(h_max, total_chunks)

    if h_min > effective_h_max:
        return []

    # Determine target number of multi-hop chunks
    target_count = max(1, total_chunks // max(1, num_multihops_factor))

    # Adjust if target is unrealistic
    if target_count * effective_h_max > total_chunks:
        target_count = total_chunks // effective_h_max

    if target_count == 0:
        return []

    rng = np.random.default_rng()

    # Generate random combinations
    indices_array = rng.choice(total_chunks, size=(target_count, effective_h_max), replace=False)
    sizes = rng.integers(low=h_min, high=effective_h_max + 1, size=target_count)

    # Create unique combinations
    unique_combos = {tuple(sorted(indices_array[i][: sizes[i]])) for i in range(target_count)}

    # Build multi-hop chunks
    return [
        MultiHopChunk(
            chunk_ids=[chunks[idx].chunk_id for idx in combo], chunks_text=[chunks[idx].chunk_text for idx in combo]
        )
        for combo in unique_combos
    ]


def run(config: dict[str, Any]) -> None:
    """
    Main entry point for the chunking pipeline stage.

    Args:
        config: Pipeline configuration dictionary
    """
    chunking_config = config.get("pipeline", {}).get("chunking", {})
    if not chunking_config.get("run", False):
        logger.info("Chunking stage is disabled. Skipping.")
        return

    logger.info("Starting chunking stage...")

    # Load dataset
    dataset = custom_load_dataset(config=config, subset="summarized")
    logger.info(f"Loaded {len(dataset)} documents for chunking")

    # Extract configuration
    params = extract_config(config)

    # Process documents
    all_single_chunks: list[list[SingleHopChunk]] = []
    all_multihop_chunks: list[list[MultiHopChunk]] = []

    start_time = time.time()

    for idx, row in enumerate(tqdm(dataset, desc="Chunking documents")):
        doc_text = row.get("document_text", "")
        doc_id = row.get("document_id", f"doc_{idx}")

        # Create single-hop chunks
        single_chunks = chunk_document(doc_text, doc_id, params.max_tokens)

        # Create multi-hop chunks
        multihop_chunks = create_multihop_chunks(
            single_chunks, params.h_min, params.h_max, params.num_multihops_factor
        )

        all_single_chunks.append(single_chunks)
        all_multihop_chunks.append(multihop_chunks)

        # Progress logging
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            logger.info(f"Progress: {idx + 1}/{len(dataset)} docs ({rate:.1f} docs/sec)")

    # Add columns to dataset
    dataset = dataset.add_column("chunks", [[asdict(chunk) for chunk in chunks] for chunks in all_single_chunks])
    dataset = dataset.add_column(
        "multihop_chunks", [[asdict(mh) for mh in multihops] for multihops in all_multihop_chunks]
    )

    # Save dataset
    hf_settings = get_hf_settings(config)

    custom_save_dataset(
        dataset=dataset,
        config=config,
        subset="chunked",
        save_local=hf_settings.local_saving,
        push_to_hub=True,
    )

    elapsed_total = time.time() - start_time
    logger.success(f"Chunking completed in {elapsed_total:.1f} seconds")
