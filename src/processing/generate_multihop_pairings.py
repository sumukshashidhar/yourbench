import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import numpy as np

from typing import List, Dict
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from collections import defaultdict

# load environment variables
load_dotenv()

BETTERBENCH_SOURCE_DATASET = os.getenv("BETTERBENCH_SOURCE_DATASET")
BETTERBENCH_MULTIHOP_PAIRINGS_DATASET = os.getenv("BETTERBENCH_MULTIHOP_PAIRINGS_DATASET")

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
    
    # Get the train split
    train_data = dataset['train']
    
    # Group by unique_document_id
    doc_groups = defaultdict(list)
    for idx, example in enumerate(train_data):
        doc_groups[example['unique_document_id']].append({
            'idx': idx,
            'chunk_uuid': example['chunk_uuid'],
            'chunk': example['chunk'],
            'chunk_location_id': example['chunk_location_id'],
            'title': example['title'],
            'document_type': example['document_type'],
            'document_name': example['document_name'],
            'summary': example['summary'],
            'unique_document_id': example['unique_document_id']
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
                'chunk_ids': [chunk['chunk_uuid'] for chunk in selected_chunks],
                'num_chunks': num_chunks,
                'chunks': [chunk['chunk'] for chunk in selected_chunks],
                'title': selected_chunks[0]['title'],  # Use title from first chunk
                'document_type': selected_chunks[0]['document_type'],
                'document_name': selected_chunks[0]['document_name'],
                'summary': selected_chunks[0]['summary'],
                'unique_document_id': doc_id
            }
            
            pairings.append(new_example)
    
    # Create new dataset
    new_dataset = Dataset.from_dict({
        'chunk_ids': [p['chunk_ids'] for p in pairings],
        'num_chunks': [p['num_chunks'] for p in pairings],
        'chunks': [p['chunks'] for p in pairings],
        'title': [p['title'] for p in pairings],
        'document_type': [p['document_type'] for p in pairings],
        'document_name': [p['document_name'] for p in pairings],
        'summary': [p['summary'] for p in pairings],
        'unique_document_id': [p['unique_document_id'] for p in pairings]
    })
    
    return new_dataset

if __name__ == "__main__":
    dataset = load_dataset(BETTERBENCH_SOURCE_DATASET)
    multihop_pairings = generate_multihop_pairings(dataset)
    # push to huggingface
    multihop_pairings.push_to_hub(BETTERBENCH_MULTIHOP_PAIRINGS_DATASET, private=True)
    pass