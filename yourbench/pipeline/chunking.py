# yourbench/pipeline/chunking.py

"""
chunking.py

Implements the Semantic Chunking stage of the YourBench pipeline using
the `intfloat/multilingual-e5-large-instruct` model for embedding-based
similarity computation. This module also generates plots of sentence-pair
similarities and saves them in the `plots/` folder for analysis.

Minimal Tracking Approach:
--------------------------
1. Single-hop chunks are stored as a list of dicts:
   [
     {
       "chunk_id": f"{document_id}_{i}",
       "chunk_text": "..."
     },
     ...
   ]
   (No chunk_uuid or chunk_location_id.)

2. Multi-hop chunks are stored as a list of dicts:
   [
     {
       "chunk_ids": ["doc_3_0", "doc_3_2"],
       "chunks_text": ["...", "..."]
     },
     ...
   ]
   (No multi_hop_id, no location indices.)

This is the minimal structure to trace which text belongs where, and how
multiple single-hop chunks are combined into multi-hop segments.
"""

import os
import random
import torch
import matplotlib.pyplot as plt

from typing import Dict, Any, List
from loguru import logger

from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

from yourbench.utils.dataset_engine import smart_load_dataset
from yourbench.utils.dataset_engine import save_dataset

import evaluate
_perplexity_metric = None
try:
    _perplexity_metric = evaluate.load("perplexity", module_type="metric", model_id="gpt2")
    logger.info("Loaded 'perplexity' metric with model_id='gpt2'.")
except Exception as e:
    logger.warning("Could not load perplexity metric from 'evaluate'. Skipping perplexity. Error: {}", e)
    _perplexity_metric = None

try:
    import textstat
    _use_textstat = True
except ImportError:
    logger.warning("Package 'textstat' not installed. Readability metrics will be skipped.")
    _use_textstat = False

os.makedirs("plots", exist_ok=True)


def run(config: Dict[str, Any]) -> None:
    chunking_cfg = config.get("pipeline", {}).get("chunking", {})
    model_name = config.get("model_roles", {}).get("chunking", ["intfloat/multilingual-e5-large-instruct"])[0]
    debug_mode = config.get("settings", {}).get("debug", False)
    if not debug_mode:
        _perplexity_metric = None
        _use_textstat = False

    if not chunking_cfg.get("run", False):
        logger.info("Chunking stage is disabled. Skipping.")
        return

    logger.info("Running chunking stage with minimal chunk tracking...")

    # 1. Load dataset
    source_dataset_name = chunking_cfg["source_dataset_name"]
    output_dataset_name = chunking_cfg["output_dataset_name"]
    dataset = smart_load_dataset(source_dataset_name, config)
    logger.debug("Loaded dataset '{}' with {} rows.", source_dataset_name, len(dataset))

    # 2. Retrieve chunking parameters
    cparams = chunking_cfg.get("chunking_configuration", {})
    l_min_tokens = cparams.get("l_min_tokens", 64)
    l_max_tokens = cparams.get("l_max_tokens", 128)
    tau_threshold = cparams.get("tau_threshold", 0.3)
    h_min = cparams.get("h_min", 2)
    h_max = cparams.get("h_max", 3)
    num_multihops_factor = cparams.get("num_multihops_factor", 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_single_hop_chunks = []
    all_multihop_chunks = []
    all_chunk_info_metrics = []
    all_similarities = []

    for idx, row in enumerate(dataset):
        doc_text = row["document_text"]
        doc_id = row.get("document_id", f"doc_{idx}")

        if not doc_text or not doc_text.strip():
            logger.warning("Document at index {} has empty text. Storing empty chunks.", idx)
            all_single_hop_chunks.append([])
            all_multihop_chunks.append([])
            all_chunk_info_metrics.append([])
            continue

        sentences = _split_into_sentences(doc_text)
        if not sentences:
            logger.warning("No valid sentences found for doc at index {}.", idx)
            all_single_hop_chunks.append([])
            all_multihop_chunks.append([])
            all_chunk_info_metrics.append([])
            continue

        # Compute embeddings for each sentence
        sentence_embeddings = _compute_embeddings(tokenizer, model, sentences, device)

        # Compute consecutive similarities
        similarities = []
        for s_i in range(len(sentences) - 1):
            cos_sim = float(F.cosine_similarity(
                sentence_embeddings[s_i].unsqueeze(0),
                sentence_embeddings[s_i + 1].unsqueeze(0),
                dim=1
            )[0])
            similarities.append(cos_sim)
        if similarities:
            all_similarities.append(similarities)

        # Single-hop chunking
        single_hop = _chunk_document(sentences, similarities, l_min_tokens, l_max_tokens, tau_threshold, doc_id)
        # Multi-hop
        multihop = _multihop_chunking(single_hop, h_min, h_max, num_multihops_factor)
        # Info metrics
        chunk_metrics = _compute_info_density_metrics(single_hop)

        all_single_hop_chunks.append(single_hop)
        all_multihop_chunks.append(multihop)
        all_chunk_info_metrics.append(chunk_metrics)

    # optional similarity plot
    if all_similarities:
        _plot_aggregated_similarities(all_similarities)

    # add columns
    dataset = dataset.add_column("chunks", all_single_hop_chunks)
    dataset = dataset.add_column("multihop_chunks", all_multihop_chunks)
    dataset = dataset.add_column("chunk_info_metrics", all_chunk_info_metrics)
    dataset = dataset.add_column("chunking_model", ["intfloat/multilingual-e5-large-instruct"] * len(dataset))

    save_dataset(dataset, "chunking", config, output_dataset_name)
    logger.success("Chunking stage complete. Dataset saved as '{}'.", output_dataset_name)


def _split_into_sentences(text: str) -> List[str]:
    import re
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    segments = re.split(r'([.!?])', text)
    sentences = []
    for i in range(0, len(segments), 2):
        if i + 1 < len(segments):
            s = (segments[i] + segments[i + 1]).strip()
        else:
            s = segments[i].strip()
        if s:
            sentences.append(s)
    return sentences


def _compute_embeddings(tokenizer, model, texts, device, max_len=512) -> List[torch.Tensor]:
    batch_dict = tokenizer(
        texts,
        max_length=max_len,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**batch_dict)
        last_hidden = outputs.last_hidden_state
        attention_mask = batch_dict["attention_mask"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
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
    Return a list of dicts: { "chunk_id": "...", "chunk_text": "..." }
    """
    chunks = []
    current_chunk = []
    current_len = 0
    chunk_index = 0

    for i in range(len(sentences)):
        s = sentences[i]
        s_len = len(s.split())

        # If adding this sentence alone exceeds l_max, store existing chunk (if any) + this alone
        if s_len >= l_max_tokens:
            if current_chunk:
                chunk_str = " ".join(current_chunk)
                chunks.append({
                    "chunk_id": f"{doc_id}_{chunk_index}",
                    "chunk_text": chunk_str
                })
                chunk_index += 1
                current_chunk = []
                current_len = 0
            # store this sentence alone
            chunks.append({
                "chunk_id": f"{doc_id}_{chunk_index}",
                "chunk_text": s
            })
            chunk_index += 1
            current_chunk = []
            current_len = 0
            continue

        current_chunk.append(s)
        current_len += s_len

        # If we've hit l_max, finalize chunk
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

        # If we have at least l_min and the similarity to next sentence is < tau, break
        if current_len >= l_min_tokens and i < len(sentences) - 1:
            if similarities[i] < tau:
                chunk_str = " ".join(current_chunk)
                chunks.append({
                    "chunk_id": f"{doc_id}_{chunk_index}",
                    "chunk_text": chunk_str
                })
                chunk_index += 1
                current_chunk = []
                current_len = 0

    # leftover
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
    Return a list of dicts: { "chunk_ids": [...], "chunks_text": [...] }
    Minimal structure to see which single-hop chunks are combined.
    """
    if not single_hop_chunks:
        return []

    multihop_groups = []
    num_multihops = max(1, len(single_hop_chunks) // num_multihops_factor)

    for _ in range(num_multihops):
        k = random.randint(h_min, h_max)
        k = min(k, len(single_hop_chunks))
        sampled_indices = sorted(random.sample(range(len(single_hop_chunks)), k))
        chosen = [single_hop_chunks[idx] for idx in sampled_indices]

        group_dict = {
            "chunk_ids": [c["chunk_id"] for c in chosen],
            "chunks_text": [c["chunk_text"] for c in chosen]
        }
        multihop_groups.append(group_dict)

    return multihop_groups


def _compute_info_density_metrics(chunks: List[Dict[str, str]]) -> List[Dict[str, float]]:
    """
    Compute per-chunk text stats (token_count, perplexity, etc.)
    Indexed the same order as 'chunks'.
    """
    results = []
    for ch in chunks:
        ctext = ch["chunk_text"]
        tokens = ctext.strip().split()
        token_count = len(tokens)
        m = {"token_count": float(token_count)}

        if token_count > 0:
            unique_toks = len(set(t.lower() for t in tokens))
            m["unique_token_ratio"] = float(unique_toks / token_count)
        else:
            m["unique_token_ratio"] = 0.0

        if token_count > 1:
            bigrams = []
            for i in range(len(tokens) - 1):
                bigrams.append((tokens[i].lower(), tokens[i+1].lower()))
            unique_bigrams = len(set(bigrams))
            m["bigram_diversity"] = float(unique_bigrams / len(bigrams))
        else:
            m["bigram_diversity"] = 0.0

        ppl_score = 0.0
        if _perplexity_metric and token_count > 0:
            try:
                result = _perplexity_metric.compute(data=[ctext], batch_size=1)
                ppl_score = result.get("mean_perplexity", 0.0)
            except Exception as e:
                logger.warning("Could not compute perplexity for chunk. Error: {}", e)
        m["perplexity"] = ppl_score

        if token_count > 0:
            avg_len = sum(len(t) for t in tokens) / token_count
            m["avg_token_length"] = float(avg_len)
        else:
            m["avg_token_length"] = 0.0

        if _use_textstat and ctext.strip():
            try:
                flesch = textstat.flesch_reading_ease(ctext)
                fog = textstat.gunning_fog(ctext)
            except Exception as e:
                logger.warning("Textstat error: {}", e)
                flesch, fog = 0.0, 0.0
            m["flesch_reading_ease"] = float(flesch)
            m["gunning_fog"] = float(fog)
        else:
            m["flesch_reading_ease"] = 0.0
            m["gunning_fog"] = 0.0

        results.append(m)
    return results


def _plot_aggregated_similarities(all_similarities: List[List[float]]) -> None:
    plt.figure(figsize=(10, 6))
    max_len = max(len(sims) for sims in all_similarities)
    avg_sim, std_sim, counts = [], [], []
    for pos in range(max_len):
        vals = [s[pos] for s in all_similarities if pos < len(s)]
        if vals:
            mean_v = sum(vals) / len(vals)
            avg_sim.append(mean_v)
            std_sim.append((sum((v - mean_v)**2 for v in vals)/len(vals))**0.5)
            counts.append(len(vals))
        else:
            break

    x = list(range(len(avg_sim)))
    plt.plot(x, avg_sim, 'b-', label='Avg Similarity')
    lower = [max(0, a - s) for a, s in zip(avg_sim, std_sim)]
    upper = [min(1, a + s) for a, s in zip(avg_sim, std_sim)]
    plt.fill_between(x, lower, upper, alpha=0.3, color='blue')
    max_count = max(counts) if counts else 1
    sizes = [30 * (c / max_count) for c in counts]
    plt.scatter(x, avg_sim, s=sizes, alpha=0.5, color='navy')
    plt.title("Average Sentence Similarity Across Documents")
    plt.xlabel("Sentence Pair Index")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plot_path = os.path.join("plots", "aggregated_similarities.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved aggregated similarity plot at '{}'", plot_path)
