# ==============================================
# yourbench/pipeline/deduplicate_multi_hop_questions.py
# ==============================================
"""
Deduplicate Multi-Hop Questions Stage

Task:
-----
Read the multi-hop question dataset, embed (question + answer) pairs,
cluster them using a similarity threshold, and pick cluster centroids
to remove near-duplicates. Finally, retain only approximately `retain_ratio`
of the dataset (or fewer) by random sampling if needed.

Approach:
---------
1. Load multi-hop question dataset from the configured subset.
2. Compute embeddings for each row's "question + self_answer" text.
3. Cluster with DBSCAN, converting "similarity_threshold" to a distance
   threshold eps = 1 - similarity_threshold.
4. From each cluster, pick the centroid item. All other items in that
   cluster are considered duplicates.
5. If we have more centroid items than `retain_ratio * total_count`,
   randomly sample them to meet the ratio requirement.
6. Save the resulting deduplicated dataset to the new subset.

Notes:
------
- Same approach as for single-hop questions, just a different source/target subset.
- We embed text with "intfloat/multilingual-e5-large-instruct".
- Keep minimal changes to the existing pipeline structure.

Configuration Example:
----------------------
deduplicate_multi_hop_questions:
  run: true
  source_subset: multi_hop_questions
  output_subset: multi_hop_questions_deduplicated
  similarity_threshold: 0.85
  retain_ratio: 0.8
"""

import math
import random
from typing import Any, Dict, List

import numpy as np
from loguru import logger
from sklearn.cluster import DBSCAN

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from yourbench.utils.dataset_engine import (
    save_dataset,
    smart_load_dataset,
)


# === Data Loading & Embedding ===


def run(config: Dict[str, Any]) -> None:
    """
    Main entry point for deduplicating multi-hop questions.

    Reads config from config["pipeline"]["deduplicate_multi_hop_questions"].
    If run=True, loads the multi-hop question dataset, clusters duplicates,
    and saves a deduplicated version.
    """
    stage_cfg = config.get("pipeline", {}).get("deduplicate_multi_hop_questions", {})
    if not stage_cfg.get("run", False):
        logger.info("Stage 'deduplicate_multi_hop_questions' is disabled. Skipping.")
        return

    source_subset = stage_cfg.get("source_subset", "multi_hop_questions")
    output_subset = stage_cfg.get("output_subset", "multi_hop_questions_deduplicated")
    similarity_threshold: float = stage_cfg.get("similarity_threshold", 0.8)
    retain_ratio: float = stage_cfg.get("retain_ratio", 0.8)

    logger.info("Starting deduplication for multi-hop questions. Source subset='{}'", source_subset)

    # 1) Load dataset
    dataset_name = config.get("hf_configuration", {}).get("global_dataset_name", "yourbench_dataset")
    ds = smart_load_dataset(dataset_name, config, source_subset)
    logger.info("Loaded multi-hop question dataset with {} rows.", len(ds))

    if not len(ds):
        logger.warning("Dataset is empty, nothing to deduplicate.")
        return

    # We'll embed "question + self_answer"
    texts = []
    for row in ds:
        q = row.get("question", "")
        a = row.get("self_answer", "")
        combined = (q + " " + a).strip()
        texts.append(combined)

    # 2) Embedding
    embeddings = _compute_embeddings(texts)
    logger.info("Computed embeddings for {} items.", len(embeddings))

    # 3) DBSCAN Clustering
    eps_value = 1.0 - similarity_threshold
    logger.debug(
        "Clustering with DBSCAN: eps={} (derived from similarity_threshold={})", eps_value, similarity_threshold
    )

    if eps_value < 0:
        logger.warning("similarity_threshold cannot exceed 1.0; adjusting eps to 0.0")
        eps_value = 0.0

    if eps_value > 1:
        logger.warning("similarity_threshold cannot be < 0.0; adjusting eps to 1.0")
        eps_value = 1.0

    clusterer = DBSCAN(eps=eps_value, min_samples=1, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)
    logger.info("DBSCAN produced {} clusters.", len(set(labels)))

    # 4) Find cluster centroids
    cluster_to_indices = {}
    for i, lbl in enumerate(labels):
        cluster_to_indices.setdefault(lbl, []).append(i)

    chosen_indices = []
    for lbl, idx_list in cluster_to_indices.items():
        if len(idx_list) == 1:
            chosen_indices.append(idx_list[0])
            continue
        chosen_idx = _find_centroid(embeddings, idx_list)
        chosen_indices.append(chosen_idx)

    logger.info("Chose {} centroid items from all clusters.", len(chosen_indices))

    # 5) Possibly sample down for ratio
    total_count = len(ds)
    desired_count = int(math.ceil(retain_ratio * total_count))
    logger.info("We want about {} items out of {}", desired_count, total_count)

    if len(chosen_indices) <= desired_count:
        final_indices = chosen_indices
        logger.info("Centroids (count={}) <= desired_count => keeping them all.", len(final_indices))
    else:
        final_indices = random.sample(chosen_indices, desired_count)
        logger.info(
            "Randomly sampled {} items from {} centroids to meet ratio {}.",
            len(final_indices),
            len(chosen_indices),
            retain_ratio,
        )

    final_indices_set = set(final_indices)
    sorted_final_indices = sorted(list(final_indices_set))
    new_ds = ds.select(sorted_final_indices)
    logger.info("Final deduplicated dataset has {} rows.", len(new_ds))

    # 6) Save
    save_dataset(
        dataset=new_ds,
        step_name="deduplicate_multi_hop_questions",
        config=config,
        output_dataset_name=dataset_name,
        output_subset=output_subset,
    )
    logger.success("Deduplication for multi-hop questions completed successfully.")


# === Helper Functions ===


def _compute_embeddings(text_list: List[str]) -> np.ndarray:
    """
    Embed each string in text_list using 'intfloat/multilingual-e5-large-instruct'
    and return a 2D numpy array of shape (n, embedding_dim).
    """
    model_name = "intfloat/multilingual-e5-large-instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_embeds = []
    batch_size = 32

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"]
            mask_expanded = mask.unsqueeze(-1).expand(outputs.size()).float()
            sum_embeddings = torch.sum(outputs * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            norm_embeddings = F.normalize(mean_embeddings, p=2, dim=1)
        all_embeds.append(norm_embeddings.cpu().numpy())

    return np.concatenate(all_embeds, axis=0)


def _find_centroid(embeds: np.ndarray, idx_list: List[int]) -> int:
    """
    Among the given indices idx_list, return the index whose embedding has
    the smallest average distance to others in that cluster.
    """
    subset = embeds[idx_list]
    dists = _pairwise_euclidean(subset)
    avg_dist = dists.mean(axis=1)
    min_idx = int(np.argmin(avg_dist))
    return idx_list[min_idx]


def _pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance among rows in x.
    x shape: (N, dim)
    output shape: (N, N)
    """
    dot = x @ x.T
    norm_sq = np.sum(x**2, axis=1, keepdims=True)
    dists = norm_sq + norm_sq.T - 2 * dot
    dists[dists < 0] = 0
    return np.sqrt(dists)
