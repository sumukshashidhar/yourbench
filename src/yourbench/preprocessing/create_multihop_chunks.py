import random
from collections import defaultdict

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset


def generate_multihop_pairings(dataset: DatasetDict) -> Dataset:
    """
    Generate multihop pairings from the dataset by grouping chunks from the same document
    and creating random combinations of 2-5 chunks.

    Args:
        dataset (DatasetDict): The input dataset containing document chunks

    Returns:
        Dataset: A new dataset with the multihop pairings
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Group by unique_document_id
    doc_groups = defaultdict(list)
    for idx, example in enumerate(dataset):
        doc_groups[example['id']].append({
            'chunk': example['chunk'],
            'chunk_location_id': example['chunk_location_id'],
            'title': example['title'],
            'summary': example['summary'],
        })

    # Generate pairings
    pairings = []

    for doc_id, chunks in doc_groups.items():
        # Skip documents with less than 2 chunks
        if len(chunks) < 2:
            continue

        # Calculate number of combinations to generate based on number of chunks
        n_chunks = len(chunks)
        n_combinations = max(1, int(np.sqrt(n_chunks)))  # Using square root as a heuristic

        # Generate combinations for different numbers of chunks (2-5)
        for _ in range(n_combinations):
            # Randomly choose number of chunks (2-5)
            num_chunks = random.randint(2, min(5, n_chunks))

            # Randomly select chunks
            selected_chunks = random.sample(chunks, num_chunks)

            # Sort by chunk_location_id to maintain document order
            selected_chunks.sort(key=lambda x: x['chunk_location_id'])

            # Create new example
            new_example = {
                'chunk_ids': [chunk['chunk_location_id'] for chunk in selected_chunks],
                'num_chunks': num_chunks,
                'chunks': [chunk['chunk'] for chunk in selected_chunks],
                'title': selected_chunks[0]['title'],  # Use title from first chunk
                'summary': selected_chunks[0]['summary'],
                'id': doc_id
            }

            pairings.append(new_example)

    # Create new dataset
    new_dataset = Dataset.from_dict({
        'document_id': [p['id'] for p in pairings],
        'chunk_ids': [p['chunk_ids'] for p in pairings],
        'num_chunks': [p['num_chunks'] for p in pairings],
        'chunks': [p['chunks'] for p in pairings],
        'title': [p['title'] for p in pairings],
        'summary': [p['summary'] for p in pairings],
    })

    return new_dataset


def create_multihop_chunks(config: dict):
    # load the dataset
    dataset = load_dataset(config["datasets"]["chunked_doucments_dataset_name"], split="train")
    # generate the multihop pairings
    multihop_pairings = generate_multihop_pairings(dataset)
    # save the multihop pairings
    multihop_pairings.push_to_hub(config["datasets"]["multihop_pairings_dataset_name"], private=True)
    return
