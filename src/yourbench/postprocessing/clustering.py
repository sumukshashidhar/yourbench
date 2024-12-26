import math
from typing import Any, Dict, List

import faiss
import numpy as np
from datasets import Dataset, concatenate_datasets
from loguru import logger
from sentence_transformers import SentenceTransformer


def _embed_dataset(
    dataset: Dataset, model_name: str, alpha: float, batch_size: int, device: str = None
) -> Dataset:
    """
    Embed the 'question' and 'answer' columns in batches, store them as 'combined_emb',
    and return the updated dataset. We do a linear combination of question/answer embeddings
    with parameter alpha.
    """
    logger.info(f"Embedding dataset with model {model_name} (alpha={alpha})")
    logger.debug(
        f"Dataset size: {len(dataset)}, batch_size: {batch_size}, device: {device}"
    )
    model = SentenceTransformer(model_name, device=device)

    def _compute_embeddings(batch):
        logger.debug(f"Processing batch of size {len(batch['question'])}")
        q_emb = model.encode(
            batch["question"],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        a_emb = model.encode(
            batch["answer"],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        logger.debug(f"Computed embeddings shapes: Q={q_emb.shape}, A={a_emb.shape}")
        return {"combined_emb": alpha * q_emb + (1 - alpha) * a_emb}

    # Use .map(...) in batched mode
    dataset = dataset.map(
        _compute_embeddings,
        batched=True,
        batch_size=batch_size,
    )
    return dataset


def _build_faiss_index(embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    """
    Build a FAISS index for L2 similarity search.
    If `use_gpu` is True and we have a GPU, then we move index to GPU.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def _cluster_via_faiss_threshold(
    embeddings: np.ndarray,
    distance_threshold: float,
    top_k: int = 50,
    use_gpu: bool = False,
) -> Dict[int, List[int]]:
    """
    A simple "threshold-based" approach to cluster:
      - For each vector i, retrieve top_k neighbors, filter by distance_threshold,
      - Then union them into clusters.

    Returns a dict: representative_idx -> list_of_indices_in_that_cluster.
    We'll refine the "representative" to mean the smallest idx, or we'll keep that
    for now and do centroid-based representative selection later.

    This is a naive approach that can work for moderate-scale data. For 100k data,
    top_k=50 might be enough. Adjust if you have more or less repetition.
    """
    logger.info(
        f"Starting clustering with threshold={distance_threshold}, top_k={top_k}"
    )
    logger.debug(f"Input embeddings shape: {embeddings.shape}")

    index = _build_faiss_index(embeddings, use_gpu=use_gpu)
    logger.debug("FAISS index built successfully")

    distances, neighbors = index.search(embeddings, top_k)
    logger.debug("Nearest neighbor search completed")

    # We'll keep track of which cluster each point belongs to
    # -1 means unassigned so far
    cluster_assignment = [-1] * len(embeddings)
    current_cluster_id = 0

    clusters = {}  # cluster_id -> list of indices

    for i in range(len(embeddings)):
        if i % 1000 == 0:  # Log progress periodically
            logger.debug(f"Processing vector {i}/{len(embeddings)}")

        if cluster_assignment[i] != -1:
            # Already assigned to a cluster
            continue

        # Start a new cluster with i
        clusters[current_cluster_id] = [i]
        cluster_assignment[i] = current_cluster_id

        # Check neighbors within threshold
        for j_idx in range(top_k):
            nb = neighbors[i, j_idx]
            dist = distances[i, j_idx]
            if nb == i:
                continue
            if dist <= distance_threshold**2:  # FAISS's IndexFlatL2 uses L2 distance
                # Assign neighbor to the same cluster
                if cluster_assignment[nb] == -1:
                    cluster_assignment[nb] = current_cluster_id
                    clusters[current_cluster_id].append(nb)

        current_size = len(clusters[current_cluster_id])
        if current_size > 1:
            logger.debug(
                f"Cluster {current_cluster_id} formed with {current_size} elements"
            )

        current_cluster_id += 1

    logger.info(f"Clustering complete. Found {len(clusters)} clusters")
    return clusters


def _compute_weights_and_reps(
    clusters: Dict[int, List[int]],
    embeddings: np.ndarray,
    lambda_val: float,
    w_max: float,
) -> Dict[int, Dict[str, Any]]:
    """
    For each cluster, compute:
      - size
      - weight = min(lambda_val * log(size), w_max)
      - representative_idx (closest to centroid)

    Returns cluster_info dict keyed by cluster_id, each value a dict:
      {
        'size': int,
        'weight': float,
        'indices': [...],
        'rep_idx': int
      }
    """
    logger.info("Computing cluster weights and representatives")
    logger.debug(
        f"Processing {len(clusters)} clusters with λ={lambda_val}, w_max={w_max}"
    )

    cluster_info = {}

    for c_id, indices in clusters.items():
        size = len(indices)
        weight = min(lambda_val * math.sqrt(size), w_max)

        if size > 1:
            logger.debug(f"Cluster {c_id}: size={size}, weight={weight:.3f}")

        # Compute centroid
        cluster_embs = embeddings[indices]
        centroid = cluster_embs.mean(axis=0, keepdims=True)  # shape = (1, dim)

        # Distances to centroid
        dists = ((cluster_embs - centroid) ** 2).sum(axis=1)
        rep_local_idx = np.argmin(dists)
        rep_idx = indices[rep_local_idx]

        cluster_info[c_id] = {
            "size": size,
            "weight": weight,
            "indices": indices,
            "rep_idx": rep_idx,
        }
    logger.info("Finished computing cluster weights and representatives")
    return cluster_info


def _assign_cluster_columns(
    dataset: Dataset, cluster_info: Dict[int, Dict[str, Any]]
) -> Dataset:
    """
    Creates two new columns in the dataset:
      - cluster_id
      - weight
    We'll do this by reversing the cluster_info.
    """
    # We need a quick lookup: which cluster does a given row index belong to?
    # And what's the cluster's weight?
    n = len(dataset)
    cluster_id_for = [-1] * n
    weight_for = [0.0] * n

    for c_id, info in cluster_info.items():
        w = info["weight"]
        for idx in info["indices"]:
            cluster_id_for[idx] = c_id
            weight_for[idx] = w

    # Now add these as columns.
    dataset = dataset.add_column("cluster_id", cluster_id_for)
    dataset = dataset.add_column("weight", weight_for)
    return dataset


def _select_representatives(
    dataset: Dataset, cluster_info: Dict[int, Dict[str, Any]]
) -> Dataset:
    """
    Build a new dataset containing only the representative row from each cluster.
    We'll copy the cluster's weight as well.
    """
    rep_rows = []

    # Create a log file to store cluster details
    with open("logs/cluster_qa_pairs.txt", "w", encoding="utf-8") as f:
        for c_id, info in cluster_info.items():
            # Convert numpy.int64 to regular Python int
            rep_idx = int(info["rep_idx"])
            row_dict = dict(dataset[rep_idx])
            row_dict["cluster_id"] = c_id
            row_dict["weight"] = info["weight"]
            rep_rows.append(row_dict)

            # Log the cluster information
            f.write(
                f"\nCluster {c_id} (size: {info['size']}, weight: {info['weight']}):\n"
            )
            f.write("-" * 80 + "\n")

            # Log representative Q&A
            f.write("REPRESENTATIVE:\n")
            f.write(f"Q: {dataset[rep_idx]['question']}\n")
            f.write(f"A: {dataset[rep_idx]['answer']}\n\n")

            # Log other members of the cluster
            f.write("SIMILAR ITEMS:\n")
            for idx in info["indices"]:
                idx = int(idx)
                if idx != rep_idx:
                    f.write(f"Q: {dataset[idx]['question']}\n")
                    f.write(f"A: {dataset[idx]['answer']}\n")
            f.write("=" * 80 + "\n")

    return Dataset.from_list(rep_rows)


def cluster_and_dedupe(dataset, config):
    logger.info("Starting clustering and deduplication process")
    logger.debug(
        f"Configuration: {config['pipeline']['reweight_and_deduplicate_questions']['cluster_configuration']}"
    )

    # Create logs directory if it doesn't exist
    import os

    os.makedirs("logs", exist_ok=True)

    # extract cluster configuration
    cluster_config = config["pipeline"]["reweight_and_deduplicate_questions"][
        "cluster_configuration"
    ]
    model_name = cluster_config["model_name"]
    alpha = cluster_config["alpha"]
    batch_size = cluster_config["batch_size"]
    distance_threshold = cluster_config["distance_threshold"]
    top_k = cluster_config["top_k"]
    lambda_val = cluster_config["lambda_val"]
    w_max = cluster_config["w_max"]

    # Get all unique complexities
    all_complexities = list(set(dataset["question_complexity"]))
    logger.info(f"Found {len(all_complexities)} unique complexity levels")

    # We'll collect the final representative datasets here
    rep_subsets = []

    for complexity_val in all_complexities:
        logger.info(f"Processing complexity level: {complexity_val}")
        subset = dataset.filter(
            lambda row: row["question_complexity"] == complexity_val
        )
        logger.debug(f"Subset size for complexity {complexity_val}: {len(subset)}")

        # === 1) Embed subset ===
        subset = _embed_dataset(subset, model_name, alpha, batch_size)
        embeddings = np.vstack(subset["combined_emb"]).astype("float32")

        # === 2) Cluster subset ===
        clusters = _cluster_via_faiss_threshold(
            embeddings=embeddings,
            distance_threshold=distance_threshold,
            top_k=top_k,
            use_gpu=True,
        )

        # === 3) Compute rep/weights ===
        cluster_info = _compute_weights_and_reps(
            clusters=clusters, embeddings=embeddings, lambda_val=lambda_val, w_max=w_max
        )

        # === 4) Assign columns & select reps ===
        combined_ds_with_clusters = _assign_cluster_columns(subset, cluster_info)
        rep_dataset = _select_representatives(combined_ds_with_clusters, cluster_info)
        # Remove embedding columns, cluster_id
        rep_dataset = rep_dataset.remove_columns(["combined_emb", "cluster_id"])

        logger.debug(f"Found {len(clusters)} clusters for complexity {complexity_val}")
        logger.debug(f"Selected {len(rep_dataset)} representatives")

        rep_subsets.append(rep_dataset)

    # Now concatenate all representative subsets from each complexity
    final_rep_dataset = concatenate_datasets(rep_subsets)
    logger.info(f"Clustering complete. Final dataset size: {len(final_rep_dataset)}")
    return final_rep_dataset
