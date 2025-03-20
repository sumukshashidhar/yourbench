# =============================================================================
# chunking.py
# =============================================================================
"""
@module chunking
@author @sumukshashidhar

This module implements the "Semantic Chunking" stage of the YourBench pipeline.
It takes ingested and optionally summarized documents, partitions them into
multiple coherent segments (single-hop chunks), and optionally creates multi-hop
chunks by sampling and concatenating various single-hop segments.

Preserves semantic relationships among sentences by leveraging embeddings from a
transformer-based model (e.g., "intfloat/multilingual-e5-large-instruct"). This
stage helps downstream question generation avoid handling entire long documents
at once, improving coverage and reducing the risk of overlooking important but
less prominent content.

Usage:
------
Typically, you do not call this module directly. Instead, the `handler.py`
automatically invokes `run(config)` if the corresponding pipeline setting
(`pipeline.chunking.run`) is enabled.

The `run(config)` function:
1. Loads a dataset specified by the pipeline configuration.
2. Splits each document into single-hop chunks, guided by user-defined token
   length constraints (`l_min_tokens`, `l_max_tokens`) and a similarity threshold
   (`tau_threshold`).
3. Creates multi-hop chunks by sampling subsets of single-hop chunks and
   concatenating them.
4. Computes optional readability and perplexity metrics for each chunk if debug
   mode is enabled.
5. Saves the dataset containing new columns: 
   - "chunks" (list of single-hop segments)
   - "multihop_chunks" (list of multi-hop segment groups)
   - "chunk_info_metrics" (various statistics)
   - "chunking_model" (the model used for embeddings).

Error Handling and Logging:
---------------------------
- All warnings, errors, and debugging information are logged to both the console
  and a dedicated log file at `logs/chunking.log`.
- If any critical errors occur while loading or processing data, the process
  logs the exception and attempts a graceful exit without crashing the entire
  pipeline.

"""

import os
import random
import re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from loguru import logger

from transformers import AutoTokenizer, AutoModel

from yourbench.utils.dataset_engine import (
    smart_load_dataset,
    smart_get_source_dataset_name,
    smart_get_output_dataset_name,
    smart_get_source_subset,
    smart_get_output_subset,
    save_dataset
)

import evaluate
import itertools  # <--- Added to generate all combinations for multi-hop chunking.

# === Stage-Specific Logger Configuration ===
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/chunking.log",
    level="DEBUG",
    rotation="10 MB",
    enqueue=True,
    backtrace=True,
    diagnose=True
)

try:
    # Attempt to load perplexity metric from evaluate
    _perplexity_metric = evaluate.load(
        "perplexity",
        module_type="metric",
        model_id="gpt2"
    )
    logger.info("Loaded 'perplexity' metric with model_id='gpt2'.")
except Exception as perplexity_load_error:
    logger.warning(
        "Could not load perplexity metric from 'evaluate'. Skipping perplexity. "
        "Error: {}", perplexity_load_error
    )
    _perplexity_metric = None

try:
    # Attempt to import textstat for readability metrics
    import textstat
    _use_textstat = True
except ImportError:
    logger.warning(
        "Package 'textstat' not installed. Readability metrics will be skipped."
    )
    _use_textstat = False

# Ensure local plots directory
os.makedirs("plots", exist_ok=True)


def run(config: Dict[str, Any]) -> None:
    """
    Main pipeline entry point for the chunking stage.

    Args:
        config (Dict[str, Any]): The entire pipeline configuration dictionary.

    Returns:
        None. This function saves the updated dataset containing chunked
        documents to disk or the Hugging Face Hub, based on the config.

    Raises:
        RuntimeError: If a critical error is encountered that prevents chunking.
                      The error is logged, and execution attempts a graceful exit.
    """
    # Retrieve chunking configuration from config
    chunking_config = config.get("pipeline", {}).get("chunking", {})
    if not chunking_config.get("run", False):
        logger.info("Chunking stage is disabled. Skipping.")
        return

    logger.info("Starting chunking stage...")

    # Attempt to load dataset
    try:
        source_dataset_name = smart_get_source_dataset_name("chunking", config)
        source_subset = smart_get_source_subset("chunking", config)
        output_dataset_name = smart_get_output_dataset_name("chunking", config)
        output_subset = smart_get_output_subset("chunking", config)

        dataset = smart_load_dataset(source_dataset_name, config, source_subset)
        logger.debug(
            "Loaded dataset '{}' with {} rows for chunking.",
            source_dataset_name, len(dataset)
        )
    except Exception as ds_error:
        logger.error("Failed to load dataset: {}", ds_error)
        logger.warning("Chunking stage cannot proceed. Exiting.")
        return

    # Retrieve chunking parameters
    chunking_parameters = chunking_config.get("chunking_configuration", {})
    l_min_tokens = chunking_parameters.get("l_min_tokens", 64)
    l_max_tokens = chunking_parameters.get("l_max_tokens", 128)
    tau_threshold = chunking_parameters.get("tau_threshold", 0.3)
    h_min = chunking_parameters.get("h_min", 2)
    h_max = chunking_parameters.get("h_max", 3)
    num_multihops_factor = chunking_parameters.get("num_multihops_factor", 2)

    # Check debug setting
    debug_mode: bool = config.get("settings", {}).get("debug", False)
    if not debug_mode:
        # If not debug mode, skip perplexity and readability to save time
        logger.debug("Skipping perplexity and readability metrics (debug mode off).")
        local_perplexity_metric = None
        local_use_textstat = False
    else:
        local_perplexity_metric = _perplexity_metric
        local_use_textstat = _use_textstat

    # Load chunking model
    try:
        # Extract model name from config if available
        model_name_list = config.get("model_roles", {}).get("chunking", [])
        if not model_name_list:
            logger.info(
                "No chunking model specified in config['model_roles']['chunking']. "
                "Using default 'intfloat/multilingual-e5-large-instruct'."
            )
            model_name = "intfloat/multilingual-e5-large-instruct"
        else:
            model_name = model_name_list[0]

        logger.info("Using chunking model: '{}'", model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
    except Exception as model_error:
        logger.error(
            "Error loading tokenizer/model '{}': {}",
            model_name, model_error
        )
        logger.warning("Chunking stage cannot proceed. Exiting.")
        return

    # Prepare data structures
    all_single_hop_chunks: List[List[Dict[str, str]]] = []
    all_multihop_chunks: List[List[Dict[str, List[str]]]] = []
    all_chunk_info_metrics: List[List[Dict[str, float]]] = []
    all_similarities: List[List[float]] = []

    # Process each document in the dataset
    for idx, row in enumerate(dataset):
        doc_text = row.get("document_text", "")
        doc_id = row.get("document_id", f"doc_{idx}")

        # If text is empty or missing
        if not doc_text or not doc_text.strip():
            logger.warning(
                "Document at index {} has empty text. Storing empty chunks.",
                idx
            )
            all_single_hop_chunks.append([])
            all_multihop_chunks.append([])
            all_chunk_info_metrics.append([])
            continue

        # Split the document into sentences
        sentences = _split_into_sentences(doc_text)
        if not sentences:
            logger.warning(
                "No valid sentences found for doc at index {}, doc_id={}.",
                idx, doc_id
            )
            all_single_hop_chunks.append([])
            all_multihop_chunks.append([])
            all_chunk_info_metrics.append([])
            continue

        # Compute embeddings for sentences
        sentence_embeddings = _compute_embeddings(
            tokenizer,
            model,
            texts=sentences,
            device=device,
            max_len=512
        )

        # Compute consecutive sentence similarities
        consecutive_sims: List[float] = []
        for sentence_index in range(len(sentences) - 1):
            cos_sim = float(
                F.cosine_similarity(
                    sentence_embeddings[sentence_index].unsqueeze(0),
                    sentence_embeddings[sentence_index + 1].unsqueeze(0),
                    dim=1
                )[0]
            )
            consecutive_sims.append(cos_sim)
        if consecutive_sims:
            all_similarities.append(consecutive_sims)

        # Create single-hop chunks
        single_hop_chunks = _chunk_document(
            sentences=sentences,
            similarities=consecutive_sims,
            l_min_tokens=l_min_tokens,
            l_max_tokens=l_max_tokens,
            tau=tau_threshold,
            doc_id=doc_id
        )

        # Create multi-hop chunks (modified to ensure no duplicates)
        multihop = _multihop_chunking(
            single_hop_chunks,
            h_min=h_min,
            h_max=h_max,
            num_multihops_factor=num_multihops_factor
        )

        # Compute metrics (token_count, perplexity, readability, etc.)
        chunk_metrics = _compute_info_density_metrics(
            single_hop_chunks,
            local_perplexity_metric,
            local_use_textstat
        )

        # Accumulate
        all_single_hop_chunks.append(single_hop_chunks)
        all_multihop_chunks.append(multihop)
        all_chunk_info_metrics.append(chunk_metrics)

    # Optional: Save aggregated similarity plot
    if all_similarities:
        _plot_aggregated_similarities(all_similarities)

    # Append new columns to dataset
    dataset = dataset.add_column("chunks", all_single_hop_chunks)
    dataset = dataset.add_column("multihop_chunks", all_multihop_chunks)
    dataset = dataset.add_column("chunk_info_metrics", all_chunk_info_metrics)
    dataset = dataset.add_column("chunking_model", [model_name] * len(dataset))

    # Save updated dataset
    try:
        save_dataset(dataset, "chunking", config, output_dataset_name, output_subset)
        logger.success(
            "Chunking stage complete. Dataset saved to '{}', subset '{}'.",
            output_dataset_name, output_subset
        )
    except Exception as save_error:
        logger.error(
            "Failed to save chunked dataset for doc '{}': {}",
            output_dataset_name, save_error
        )


def _split_into_sentences(text: str) -> List[str]:
    """
    Splits the input text into sentences using a simple rule-based approach
    that looks for punctuation delimiters ('.', '!', '?').

    Args:
        text (str): The full document text to be split.

    Returns:
        List[str]: A list of sentence strings.
    """
    # Replace newlines with spaces for consistency
    normalized_text = text.replace("\n", " ").strip()
    if not normalized_text:
        return []

    # Split using capturing parentheses to retain delimiters, then recombine.
    segments = re.split(r'([.!?])', normalized_text)
    sentences: List[str] = []
    for i in range(0, len(segments), 2):
        if i + 1 < len(segments):
            # Combine the text and delimiter
            candidate = (segments[i] + segments[i + 1]).strip()
        else:
            # If no delimiter segment, use the text directly
            candidate = segments[i].strip()
        if candidate:
            sentences.append(candidate)
    return sentences


def _compute_embeddings(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    texts: List[str],
    device: torch.device,
    max_len: int = 512
) -> List[torch.Tensor]:
    """
    Computes sentence embeddings by mean pooling the last hidden states,
    normalized to unit length.

    Args:
        tokenizer (AutoTokenizer): A Hugging Face tokenizer.
        model (AutoModel): A pretrained transformer model to generate embeddings.
        texts (List[str]): The list of sentence strings to be embedded.
        device (torch.device): The device on which to run inference (CPU or GPU).
        max_len (int): Max sequence length for tokenization.

    Returns:
        List[torch.Tensor]: A list of PyTorch tensors (one per sentence).
    """
    if not texts:
        return []

    batch_dict = tokenizer(
        texts,
        max_length=max_len,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**batch_dict)
        last_hidden_states = outputs.last_hidden_state
        attention_mask = batch_dict["attention_mask"]

        # Zero out non-attended tokens
        last_hidden_states = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )

        # Mean pooling
        sum_hidden = last_hidden_states.sum(dim=1)
        valid_token_counts = attention_mask.sum(dim=1, keepdim=True)
        embeddings = sum_hidden / valid_token_counts.clamp(min=1e-9)

        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return list(embeddings.cpu())


def _chunk_document(
    sentences: List[str],
    similarities: List[float],
    l_min_tokens: int,
    l_max_tokens: int,
    tau: float,
    doc_id: str
) -> List[Dict[str, str]]:
    """
    Creates single-hop chunks from sentences, ensuring each chunk is at least
    l_min_tokens in length and at most l_max_tokens, and introducing a chunk
    boundary when consecutive sentence similarity is below a threshold tau.

    Args:
        sentences (List[str]): The list of sentences for a single document.
        similarities (List[float]): Cosine similarities between consecutive sentences.
        l_min_tokens (int): Minimum tokens per chunk.
        l_max_tokens (int): Maximum tokens per chunk.
        tau (float): Similarity threshold for introducing a chunk boundary.
        doc_id (str): Unique identifier for the document.

    Returns:
        List[Dict[str, str]]: A list of chunk dictionaries with keys:
          - "chunk_id": A string representing the chunk identifier.
          - "chunk_text": The content of the chunk.
    """
    chunks: List[Dict[str, str]] = []
    current_chunk: List[str] = []
    current_len: int = 0
    chunk_index: int = 0

    for i, sentence in enumerate(sentences):
        sentence_token_count = len(sentence.split())

        # If one sentence alone exceeds l_max, finalize the current chunk if non-empty,
        # then store this sentence as its own chunk.
        if sentence_token_count >= l_max_tokens:
            # Dump the current chunk
            if current_chunk:
                chunk_str = " ".join(current_chunk)
                chunks.append({
                    "chunk_id": f"{doc_id}_{chunk_index}",
                    "chunk_text": chunk_str
                })
                chunk_index += 1
                current_chunk = []
                current_len = 0
            # Store the sentence alone
            chunks.append({
                "chunk_id": f"{doc_id}_{chunk_index}",
                "chunk_text": sentence
            })
            chunk_index += 1
            continue

        # Otherwise, add this sentence to the current chunk
        current_chunk.append(sentence)
        current_len += sentence_token_count

        # If we exceed l_max, close the current chunk and start a new one
        if current_len >= l_max_tokens:
            chunk_str = " ".join(current_chunk)
            chunks.append({
                "chunk_id": f"{doc_id}_{chunk_index}",
                "chunk_text": chunk_str
            })
            chunk_index += 1
            current_chunk = []
            current_len = 0
            continue

        # If we have at least l_min tokens and the next sentence similarity is below threshold, break here
        if (current_len >= l_min_tokens) and (i < len(sentences) - 1):
            if similarities[i] < tau:
                chunk_str = " ".join(current_chunk)
                chunks.append({
                    "chunk_id": f"{doc_id}_{chunk_index}",
                    "chunk_text": chunk_str
                })
                chunk_index += 1
                current_chunk = []
                current_len = 0

    # Any leftover
    if current_chunk:
        chunk_str = " ".join(current_chunk)
        chunks.append({
            "chunk_id": f"{doc_id}_{chunk_index}",
            "chunk_text": chunk_str
        })

    return chunks


def _multihop_chunking(
    single_hop_chunks: List[Dict[str, str]],
    h_min: int,
    h_max: int,
    num_multihops_factor: int
) -> List[Dict[str, List[str]]]:
    """
    Creates multi-hop chunks by generating all valid combinations of single-hop chunks
    (from size h_min to h_max), then shuffling and picking the desired number. This
    ensures no repeated multi-hop chunk grouping is created.

    The total multi-hop chunks to select is determined by:
        num_multihops = max(1, total_single_hops // num_multihops_factor).

    If the number of possible unique combinations is less than or equal to num_multihops,
    we return all. Otherwise, we select a random sample of size num_multihops from those
    unique combinations.

    Args:
        single_hop_chunks (List[Dict[str, str]]): List of single-hop chunk dicts.
        h_min (int): Minimum number of chunks to combine.
        h_max (int): Maximum number of chunks to combine.
        num_multihops_factor (int): Determines how many multi-hop chunks to generate,
                                    typically a fraction of the total single-hop chunks.

    Returns:
        List[Dict[str, List[str]]]: Each element has:
          - "chunk_ids": The list of chunk_ids used in this multi-hop chunk.
          - "chunks_text": The list of actual chunk texts.
    """
    if not single_hop_chunks:
        return []

    total_single_hops = len(single_hop_chunks)
    # This is our target count for how many multi-hop combos we want to keep
    num_multihops = max(1, total_single_hops // num_multihops_factor)

    # Build a list of ALL possible multi-hop combinations from h_min to h_max
    all_combos: List[Dict[str, List[str]]] = []
    for size in range(h_min, h_max + 1):
        if size > total_single_hops:
            break
        for combo_indices in itertools.combinations(range(total_single_hops), size):
            chosen_chunks = [single_hop_chunks[idx] for idx in combo_indices]
            group_dict = {
                "chunk_ids": [c["chunk_id"] for c in chosen_chunks],
                "chunks_text": [c["chunk_text"] for c in chosen_chunks]
            }
            all_combos.append(group_dict)

    random.shuffle(all_combos)
    if len(all_combos) <= num_multihops:
        return all_combos
    else:
        return all_combos[:num_multihops]


def _compute_info_density_metrics(
    chunks: List[Dict[str, str]],
    local_perplexity_metric: Optional[Any],
    local_use_textstat: bool
) -> List[Dict[str, float]]:
    """
    Computes optional statistics for each chunk, including token count, perplexity,
    readability (flesch, gunning fog), and basic lexical diversity metrics.

    Args:
        chunks (List[Dict[str, str]]): The list of chunk dictionaries produced by
                                       _chunk_document.
        local_perplexity_metric (Optional[Any]): If provided, used to compute
                                                 perplexity (from evaluate.load("perplexity")).
        local_use_textstat (bool): If True, compute text readability metrics using textstat.

    Returns:
        List[Dict[str, float]]: One dictionary per chunk with fields like:
          - "token_count"
          - "unique_token_ratio"
          - "bigram_diversity"
          - "perplexity"
          - "avg_token_length"
          - "flesch_reading_ease"
          - "gunning_fog"
    """
    results: List[Dict[str, float]] = []
    for chunk in chunks:
        chunk_text: str = chunk["chunk_text"]
        tokens = chunk_text.strip().split()
        token_count: int = len(tokens)

        metrics: Dict[str, float] = {}
        metrics["token_count"] = float(token_count)

        if token_count > 0:
            unique_toks = len(set(t.lower() for t in tokens))
            metrics["unique_token_ratio"] = float(unique_toks / token_count)
        else:
            metrics["unique_token_ratio"] = 0.0

        # Bigram diversity
        if token_count > 1:
            bigrams = []
            for i in range(token_count - 1):
                bigrams.append((tokens[i].lower(), tokens[i + 1].lower()))
            unique_bigrams = len(set(bigrams))
            metrics["bigram_diversity"] = float(unique_bigrams / len(bigrams))
        else:
            metrics["bigram_diversity"] = 0.0

        # Perplexity
        ppl_score: float = 0.0
        if local_perplexity_metric and token_count > 0:
            try:
                result = local_perplexity_metric.compute(data=[chunk_text], batch_size=1)
                ppl_score = result.get("mean_perplexity", 0.0)
            except Exception as e:
                logger.warning("Could not compute perplexity for chunk. Error: {}", e)
        metrics["perplexity"] = ppl_score

        # Average token length
        if token_count > 0:
            avg_len = sum(len(t) for t in tokens) / token_count
            metrics["avg_token_length"] = float(avg_len)
        else:
            metrics["avg_token_length"] = 0.0

        # Readability
        if local_use_textstat and chunk_text.strip():
            try:
                flesch = textstat.flesch_reading_ease(chunk_text)
                fog = textstat.gunning_fog(chunk_text)
                metrics["flesch_reading_ease"] = float(flesch)
                metrics["gunning_fog"] = float(fog)
            except Exception as e:
                logger.warning("Textstat error: {}", e)
                metrics["flesch_reading_ease"] = 0.0
                metrics["gunning_fog"] = 0.0
        else:
            metrics["flesch_reading_ease"] = 0.0
            metrics["gunning_fog"] = 0.0

        results.append(metrics)

    return results


def _plot_aggregated_similarities(all_similarities: List[List[float]]) -> None:
    """
    Plots the average cosine similarity for each sentence-pair position across
    all documents, with shaded regions representing one standard deviation.

    Args:
        all_similarities (List[List[float]]): A list of lists, where each
            sub-list is the array of consecutive sentence similarities for
            a particular document.
    """
    if not all_similarities:
        logger.debug("No similarities to plot. Skipping aggregated similarity plot.")
        return

    plt.figure(figsize=(10, 6))
    max_len = max(len(sims) for sims in all_similarities)

    avg_sim: List[float] = []
    std_sim: List[float] = []
    counts: List[int] = []

    for position in range(max_len):
        vals = [
            s[position] for s in all_similarities if position < len(s)
        ]
        if vals:
            mean_val = sum(vals) / len(vals)
            variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
            stddev_val = variance ** 0.5

            avg_sim.append(mean_val)
            std_sim.append(stddev_val)
            counts.append(len(vals))
        else:
            break

    # X-axis positions
    x_positions = list(range(len(avg_sim)))
    plt.plot(x_positions, avg_sim, 'b-', label='Avg Similarity')

    # Create confidence interval region
    lower_bound = [max(0, a - s) for a, s in zip(avg_sim, std_sim)]
    upper_bound = [min(1, a + s) for a, s in zip(avg_sim, std_sim)]
    plt.fill_between(x_positions, lower_bound, upper_bound, alpha=0.3, color='blue')

    # Plot data points with size reflecting how many docs contributed
    max_count = max(counts) if counts else 1
    sizes = [30.0 * (c / max_count) for c in counts]
    plt.scatter(x_positions, avg_sim, s=sizes, alpha=0.5, color='navy')

    plt.title("Average Consecutive Sentence Similarity Across Documents")
    plt.xlabel("Sentence Pair Index")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plot_path: str = os.path.join("plots", "aggregated_similarities.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved aggregated similarity plot at '{}'.", plot_path)