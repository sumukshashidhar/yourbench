"""
chunking.py

Implements the Semantic Chunking stage of the YourBench pipeline using
the `intfloat/multilingual-e5-large-instruct` model for embedding-based
similarity computation. This module also generates plots of sentence-pair
similarities and saves them in the `plots/` folder for debugging and
analysis purposes.

References (from your paper):
- Section 2.2.3 on Semantic Chunking
- Equation for similarity-based boundary detection
- Multi-hop chunking approach
- Now uses E5 embeddings for semantic similarity
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
from yourbench.utils.saving_engine import save_dataset

# === GLOBALS ===
E5_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
os.makedirs("plots", exist_ok=True)  # Ensure plots folder exists for saving graphs

import evaluate
_perplexity_metric = None

try:
    # Attempt to load perplexity from the 'evaluate' library using GPT-2
    _perplexity_metric = evaluate.load("perplexity", module_type="metric", model_id="gpt2")
    logger.info("Loaded 'perplexity' metric with model_id='gpt2'.")
except Exception as e:
    logger.warning(
        "Could not load perplexity metric from 'evaluate'. "
        "We will skip perplexity-based info density. Error: {}", e
    )
    _perplexity_metric = None

# ### NEW METRICS ###
# Text readability stats. Must `pip install textstat` in your environment.
try:
    import textstat
    _use_textstat = True
except ImportError:
    logger.warning("Package 'textstat' not installed. Readability metrics will be skipped.")
    _use_textstat = False


def run(config: Dict[str, Any]) -> None:
    """
    Run the chunking stage of the pipeline.

    This function:
      1. Loads the source dataset from Hugging Face using `smart_load_dataset`.
      2. Retrieves the chunking parameters (l_min_tokens, l_max_tokens, tau_threshold, etc.) from config.
      3. For each document:
         - Splits it into sentences.
         - Computes embeddings for each sentence via the E5 model.
         - Calculates consecutive sentence similarities.
         - Determines chunk boundaries based on l_min_tokens, l_max_tokens, and tau_threshold.
         - Generates a plot of the consecutive similarities and saves it in plots/.
      4. Optionally builds multi-hop chunks by sampling subsets of the single-hop chunks.
      5. Computes "information density" metrics for each chunk (token_count, perplexity, etc.).
      6. Appends 'chunks', 'multihop_chunks', and 'chunk_info_metrics' columns to the dataset, then saves it.

    Configuration Example:
        pipeline:
          chunking:
            source_dataset_name: yb_demo_ingested_documents_with_summaries
            output_dataset_name: yb_demo_chunked_documents
            chunking_configuration:
              l_min_tokens: 64
              l_max_tokens: 128
              tau_threshold: 0.3
              h_min: 2
              h_max: 4
            run: true
    """
    chunking_cfg = config.get("pipeline", {}).get("chunking", {})
    if not chunking_cfg.get("run", False):
        logger.info("Chunking stage is disabled. Skipping.")
        return

    logger.info("Running chunking stage with E5 embeddings...")

    # === Step 1: Load dataset ===
    source_dataset_name = chunking_cfg["source_dataset_name"]
    output_dataset_name = chunking_cfg["output_dataset_name"]
    dataset = smart_load_dataset(source_dataset_name, config)
    logger.debug("Loaded dataset '{}' with {} rows.", source_dataset_name, len(dataset))

    # === Step 2: Retrieve chunking parameters ===
    cparams = chunking_cfg.get("chunking_configuration", {})
    l_min_tokens = cparams.get("l_min_tokens", 256)
    l_max_tokens = cparams.get("l_max_tokens", 1024)
    tau_threshold = cparams.get("tau_threshold", 0.3)
    h_min = cparams.get("h_min", 2)
    h_max = cparams.get("h_max", 3)

    logger.debug(
        "Chunking configuration: l_min_tokens={}, l_max_tokens={}, tau_threshold={}, h_min={}, h_max={}",
        l_min_tokens, l_max_tokens, tau_threshold, h_min, h_max
    )

    # === Load E5 model/tokenizer once (GPU-accelerated if available) ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_NAME)
    model = AutoModel.from_pretrained(E5_MODEL_NAME).to(device).eval()

    # === Step 3: Perform chunking on each row (document) ===
    all_single_hop_chunks = []
    all_multihop_chunks = []
    all_chunk_info_metrics = []

    for idx, row in enumerate(dataset):
        doc_text = row["document_text"]
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

        # === 3A: Compute embeddings for each sentence ===
        sentence_embeddings = _compute_embeddings(tokenizer, model, sentences, device)

        # === 3B: Compute consecutive similarity array ===
        similarities = []
        for s_i in range(len(sentences) - 1):
            cos_sim = float(
                F.cosine_similarity(
                    sentence_embeddings[s_i].unsqueeze(0),
                    sentence_embeddings[s_i + 1].unsqueeze(0),
                    dim=1
                )[0]
            )
            similarities.append(cos_sim)

        # === 3C: Generate a plot of these similarities for inspection ===
        _plot_sentence_similarities(similarities, idx)

        # === 3D: Single-hop chunking with boundary logic ===
        single_hop = _chunk_document(
            sentences,
            similarities,
            l_min_tokens,
            l_max_tokens,
            tau_threshold
        )

        # === 3E: Multi-hop chunking ===
        multihop = _multihop_chunking(single_hop, h_min, h_max)

        # === 3F: Compute info density metrics for each single-hop chunk
        chunk_metrics = _compute_info_density_metrics(single_hop)

        all_single_hop_chunks.append(single_hop)
        all_multihop_chunks.append(multihop)
        all_chunk_info_metrics.append(chunk_metrics)

    # === Step 4: Add new columns and save ===
    dataset = dataset.add_column("chunks", all_single_hop_chunks)
    dataset = dataset.add_column("multihop_chunks", all_multihop_chunks)
    dataset = dataset.add_column("chunk_info_metrics", all_chunk_info_metrics)
    dataset = dataset.add_column("chunking_model", [E5_MODEL_NAME] * len(dataset))

    # === Step 5: Save dataset ===
    save_dataset(dataset, "chunking", config, output_dataset_name)
    logger.success("Chunking stage complete. Dataset updated and saved as '{}'.", output_dataset_name)


def _compute_embeddings(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    texts: List[str],
    device: torch.device,
    max_len: int = 512
) -> List[torch.Tensor]:
    """
    Compute sentence embeddings using the E5 model (intfloat/multilingual-e5-large-instruct).

    Args:
        tokenizer (AutoTokenizer): The HF tokenizer for E5.
        model (AutoModel): The HF model (E5).
        texts (List[str]): List of sentences to embed.
        device (torch.device): CPU or GPU device.
        max_len (int): Tokenization truncation length (default=512).

    Returns:
        A list of torch.Tensor, each with dimension [hidden_size].
    """
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
        # zero out non-token positions
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        # average-pool
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        # normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return list(embeddings.cpu())


def _split_into_sentences(text: str) -> List[str]:
    """
    Naive sentence splitting by '.', '!', or '?'.
    For more robust usage, consider spaCy or NLTK-based splitting.
    """
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


def _chunk_document(
    sentences: List[str],
    similarities: List[float],
    l_min_tokens: int,
    l_max_tokens: int,
    tau: float
) -> List[str]:
    """
    Split a document into semantic chunks based on E5-based sentence similarities
    and token-length constraints.

    The chunk boundary rule from the paper:
      - Start a new chunk once the current chunk has at least l_min_tokens, AND
        either (i) the similarity between consecutive sentences is < tau, OR
        (ii) adding another sentence would exceed l_max_tokens.

    Implementation details:
      - We treat each sentence's length as the count of whitespace-delimited tokens.
      - similarities[i] is the sim between sentences[i] and sentences[i+1].
      - We'll iterate sentence by sentence, accumulate them in `current_chunk`,
        and watch for boundary triggers.
    """
    chunks = []
    current_chunk = []
    current_len = 0

    n_sents = len(sentences)
    for i in range(n_sents):
        s = sentences[i]
        s_tokens = s.split()
        s_len = len(s_tokens)

        # If adding this sentence alone exceeds l_max, isolate it as a single chunk
        if s_len >= l_max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            chunks.append(s)
            current_chunk = []
            current_len = 0
            continue

        current_chunk.append(s)
        current_len += s_len

        # Condition A: Exceeds l_max
        if current_len >= l_max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
            continue

        # Condition B: If we have at least l_min, check the similarity to the next sentence
        if current_len >= l_min_tokens and i < n_sents - 1:
            if similarities[i] < tau:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _multihop_chunking(
    single_hop_chunks: List[str],
    h_min: int,
    h_max: int
) -> List[str]:
    """
    Create multi-hop chunks by randomly sampling subsets of single-hop chunks
    and concatenating them. (Per Section 2.2.3 in the paper.)

    Steps:
      1. Let k ~ Uniform(h_min, h_max)
      2. Sample k distinct indices from single_hop_chunks
      3. Concatenate them in ascending order
      4. Return as a single multihop chunk in a list
    """
    if not single_hop_chunks:
        return []

    k = random.randint(h_min, h_max)
    k = min(k, len(single_hop_chunks))
    sampled_indices = sorted(random.sample(range(len(single_hop_chunks)), k))
    multi_hop_concat = " ".join(single_hop_chunks[i] for i in sampled_indices)

    return [multi_hop_concat]


def _plot_sentence_similarities(similarities: List[float], doc_idx: int) -> None:
    """
    Plot and save the distribution of consecutive sentence similarities for a given document.
    """
    if not similarities:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(similarities)), similarities, marker='o')
    plt.title(f"Consecutive Sentence Similarities (doc {doc_idx})")
    plt.xlabel("Sentence Pair Index (i -> i+1)")
    plt.ylabel("Cosine Similarity (E5 Embeddings)")
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plot_path = os.path.join("plots", f"chunking_document_{doc_idx}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.debug("Saved similarity plot for document {} at '{}'.", doc_idx, plot_path)


# -------------------------------------------------------------------
#    NEW FEATURE: Additional "Information Density" Metrics
# -------------------------------------------------------------------
def _compute_info_density_metrics(chunks: List[str]) -> List[Dict[str, float]]:
    """
    For each chunk, compute several "information density" metrics, such as:
      1. token_count
      2. unique_token_ratio
      3. bigram_diversity
      4. perplexity (if loaded)
      5. average token length
      6. readability metrics (Flesch, Gunning Fog) if textstat is installed

    Returns a list of dicts, each for one chunk, e.g.:
    [
      {
        "token_count": 120,
        "unique_token_ratio": 0.82,
        "bigram_diversity": 0.70,
        "perplexity": 54.3,
        "avg_token_length": 4.21,
        "flesch_reading_ease": 35.12,
        "gunning_fog": 14.7
      },
      ...
    ]

    If a metric cannot be computed (e.g., missing libs, short text),
    the corresponding field is set to 0.0 (or a safe default).
    """
    results = []

    for i, ctext in enumerate(chunks):
        tokens = ctext.strip().split()
        token_count = len(tokens)

        # 1. token_count
        metrics_dict = {
            "token_count": float(token_count)
        }

        # 2. unique_token_ratio
        if token_count > 0:
            unique_tokens = len(set(t.lower() for t in tokens))
            metrics_dict["unique_token_ratio"] = float(unique_tokens / token_count)
        else:
            metrics_dict["unique_token_ratio"] = 0.0

        # 3. bigram_diversity
        if token_count > 1:
            bigrams = []
            for idx_token in range(len(tokens) - 1):
                bigrams.append((tokens[idx_token].lower(), tokens[idx_token + 1].lower()))
            unique_bigrams = len(set(bigrams))
            metrics_dict["bigram_diversity"] = float(unique_bigrams / (len(bigrams)))
        else:
            metrics_dict["bigram_diversity"] = 0.0

        # 4. perplexity (if available)
        ppl_score = 0.0
        if _perplexity_metric and token_count > 0:
            try:
                ppl_result = _perplexity_metric.compute(data=[ctext], batch_size=1)
                ppl_score = ppl_result.get("mean_perplexity", 0.0)
            except Exception as e:
                logger.warning(
                    "Could not compute perplexity for chunk {}. "
                    "Likely short or invalid text. Error: {}", i, e
                )
        metrics_dict["perplexity"] = float(ppl_score)

        # 5. average token length
        if token_count > 0:
            avg_len = sum(len(t) for t in tokens) / token_count
            metrics_dict["avg_token_length"] = float(avg_len)
        else:
            metrics_dict["avg_token_length"] = 0.0

        # 6. readability metrics (if textstat is installed)
        if _use_textstat and ctext.strip():
            try:
                flesch = textstat.flesch_reading_ease(ctext)
                fog = textstat.gunning_fog(ctext)
            except Exception as e:
                logger.warning("Textstat error for chunk {}: {}", i, e)
                flesch = 0.0
                fog = 0.0
            metrics_dict["flesch_reading_ease"] = float(flesch)
            metrics_dict["gunning_fog"] = float(fog)
        else:
            metrics_dict["flesch_reading_ease"] = 0.0
            metrics_dict["gunning_fog"] = 0.0

        # Log the results for debug
        logger.debug(
            "Chunk {} => tokens={}, uniq_ratio={:.3f}, bigram_div={:.3f}, ppl={:.2f}, avg_len={:.2f}, flesch={:.2f}, fog={:.2f}",
            i,
            metrics_dict["token_count"],
            metrics_dict["unique_token_ratio"],
            metrics_dict["bigram_diversity"],
            metrics_dict["perplexity"],
            metrics_dict["avg_token_length"],
            metrics_dict["flesch_reading_ease"],
            metrics_dict["gunning_fog"]
        )

        results.append(metrics_dict)

    return results
