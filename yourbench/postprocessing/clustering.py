import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


def create_qa_representation(row: pd.Series, separator: str = " [SEP] ") -> str:
    """Create combined question-answer representation."""
    return f"{row['question']}{separator}{row['answer']}"


def compute_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32) -> np.ndarray:
    """Compute embeddings using the specified model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using device: {device}")
    model = SentenceTransformer(model_name).to(device)

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True, device=device)
        embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(embeddings)


def deduplicate_question_type(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    eps: float,
    min_samples: int,
    question_type: str,
    output_file: str,
) -> Tuple[pd.DataFrame, Dict]:
    """Deduplicate a single question type using DBSCAN, and compute sublinear weights."""
    # Ensure the directory for output_file exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    logger.info(f"Processing question type: {question_type}")
    logger.debug(f"Initial size: {len(df)} questions")

    # Store original indices
    df = df.reset_index(drop=True)  # Reset index to ensure consecutive integers

    # Perform clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    clusters = clustering.fit_predict(embeddings)

    keep_indices = []
    cluster_info = {}
    keep_index_to_weight = {}

    # Open file for writing cluster information
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Question Type: {question_type}\n")
        f.write(f"Initial size: {len(df)} questions\n")
        f.write(f"Parameters: eps={eps}, min_samples={min_samples}\n")
        f.write(f"{'=' * 80}\n\n")

        # Handle noise points (unique questions, labeled as -1)
        noise_points = np.where(clusters == -1)[0]
        keep_indices.extend(noise_points)
        # Noise points are effectively "clusters" of size 1
        for idx in noise_points:
            keep_index_to_weight[idx] = 1.0

        f.write(f"Found {len(noise_points)} unique questions\n")

        # Process actual clusters
        for label in set(clusters):
            if label == -1:
                continue

            indices = np.where(clusters == label)[0]
            if len(indices) < 2:
                continue

            # Find the central point
            cluster_embeddings = embeddings[indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = cosine_similarity([centroid], cluster_embeddings)[0]
            central_idx = indices[np.argmax(distances)]

            # Write cluster information to file
            f.write(f"\nCluster {label}:\n")
            f.write(f"Size: {len(indices)} questions\n")
            f.write("Representative QA pair:\n")
            f.write(f"Q: {df.iloc[central_idx]['question']}\n")
            f.write(f"A: {df.iloc[central_idx]['answer']}\n")
            f.write("Other cluster members:\n")
            for idx in indices:
                if idx != central_idx:
                    f.write(f"Q: {df.iloc[idx]['question']}\n")
            f.write("-" * 80 + "\n")

            # Keep the representative
            keep_indices.append(central_idx)

            # Compute sublinear weight
            cluster_size = len(indices)
            weight_value = math.sqrt(cluster_size)

            # Save cluster info
            cluster_info[label] = {
                "size": cluster_size,
                "representative_idx": central_idx,
                "member_indices": indices.tolist(),
            }

            # Assign weight to the retained index
            keep_index_to_weight[central_idx] = weight_value

    # Create deduplicated dataframe
    keep_indices = sorted(keep_indices)  # Ensure indices are sorted
    deduplicated_df = df.iloc[keep_indices].copy()

    # Reset index and add weights using the original indices
    deduplicated_df = deduplicated_df.reset_index(drop=True)
    deduplicated_df["weight"] = [keep_index_to_weight[idx] for idx in keep_indices]

    # Log summary statistics
    shrinkage = (len(df) - len(deduplicated_df)) / len(df) * 100
    logger.info(f"Deduplication results for {question_type}:")
    logger.info(f"Original: {len(df)} → Deduplicated: {len(deduplicated_df)} (Shrinkage: {shrinkage:.2f}%)")
    logger.debug(f"Number of clusters (excluding noise): {len(cluster_info)}")

    return deduplicated_df, cluster_info


def cluster_and_dedupe(dataset: Dataset, config: dict) -> Dataset:
    """
    Cluster and deduplicate the dataset based on provided configuration.
    A 'weight' column is added (sublinear in cluster size).

    Args:
        dataset: Input dataset to deduplicate
        config: Configuration dictionary containing parameters

    Returns:
        Deduplicated dataset
    """
    # Extract config values
    config_values = config["pipeline"]["reweight_and_deduplicate_questions"]["cluster_configuration"]
    embedding_model_name = config_values["model_name"]
    eps = config_values["eps"]
    min_samples = config_values["min_samples"]
    output_file = config_values.get("output_file", "deduplication_results.txt")

    logger.info("Starting deduplication process")
    logger.debug(f"Configuration: eps={eps}, min_samples={min_samples}")
    logger.debug(f"Embedding model: {embedding_model_name}")

    # Convert to pandas for processing
    df = dataset.to_pandas()

    # Process each question type separately
    all_dedup_dfs = []
    for qtype, group_df in df.groupby("question_type"):
        logger.debug(f"Starting processing for {qtype}")

        # Create combined representations
        group_df["qa_combined"] = group_df.apply(create_qa_representation, axis=1)

        # Compute embeddings
        embeddings = compute_embeddings(group_df["qa_combined"].tolist(), model_name=embedding_model_name)

        # Deduplicate
        dedup_df, _ = deduplicate_question_type(
            group_df,
            embeddings,
            eps=eps,
            min_samples=min_samples,
            question_type=qtype,
            output_file=output_file,
        )

        all_dedup_dfs.append(dedup_df)

    # Combine all deduplicated dataframes
    final_df = pd.concat(all_dedup_dfs, ignore_index=True)

    # Log final statistics
    logger.info(f"Deduplication complete: {len(df)} → {len(final_df)} questions")
    logger.info(f"Overall shrinkage: {(len(df) - len(final_df)) / len(df) * 100:.2f}%")

    # # Write final statistics to file
    # with open(output_file, "a", encoding="utf-8") as f:
    #     f.write("\nFINAL DEDUPLICATION STATISTICS\n")
    #     f.write("=" * 30 + "\n")
    #     f.write(f"Original dataset size: {len(df)}\n")
    #     f.write(f"Deduplicated dataset size: {len(final_df)}\n")
    #     f.write(
    #         f"Overall shrinkage: {(len(df) - len(final_df)) / len(df) * 100:.2f}%\n"
    #     )

    # Convert back to huggingface Dataset
    return Dataset.from_pandas(final_df)
