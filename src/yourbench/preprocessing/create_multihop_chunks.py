import random
from collections import defaultdict
from typing import Dict

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger


def generate_multihop_pairings(dataset: DatasetDict) -> Dataset:
    logger.info("Starting multihop pairings generation process")

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    logger.debug("Set random seeds for reproducibility")

    # Group by unique_document_id
    doc_groups = defaultdict(list)
    for idx, example in enumerate(dataset):
        doc_groups[example['id']].append({
            'chunk': example['chunk'],
            'chunk_location_id': example['chunk_location_id'],
            'title': example['title'],
            'summary': example['summary'],
        })
    logger.info(f"Grouped {len(doc_groups)} unique documents for pairing generation")

    # Generate pairings
    pairings = []
    logger.debug("Starting to generate chunk combinations")

    for doc_id, chunks in doc_groups.items():
        if len(chunks) < 2:
            logger.debug(f"Skipping document {doc_id} - insufficient chunks ({len(chunks)})")
            continue

        n_chunks = len(chunks)
        n_combinations = max(1, int(np.sqrt(n_chunks)))
        logger.debug(f"Generating {n_combinations} combinations for document {doc_id} with {n_chunks} chunks")

        # Calculate number of combinations to generate based on number of chunks
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

    logger.info(f"Generated {len(pairings)} multihop pairings in total")

    # Create new dataset
    logger.debug("Converting pairings to Hugging Face Dataset format")
    new_dataset = Dataset.from_dict({
        'document_id': [p['id'] for p in pairings],
        'chunk_ids': [p['chunk_ids'] for p in pairings],
        'num_chunks': [p['num_chunks'] for p in pairings],
        'chunks': [p['chunks'] for p in pairings],
        'title': [p['title'] for p in pairings],
        'summary': [p['summary'] for p in pairings],
    })
    logger.success(f"Successfully created dataset with {len(new_dataset)} multihop entries")

    return new_dataset


def get_full_dataset_name(config: Dict) -> str:
    source_dataset_name = config["selected_choices"]["make_multihop_chunks"]["source_dataset_name"]
    return config["configurations"]["hf_organization"] + "/" + source_dataset_name


def handle_dataset_push(dataset: Dataset, dataset_name: str, config: dict) -> None:

    if config["configurations"]["push_to_huggingface"]:
        privacy = False if config["configurations"]["set_hf_repo_visibility"] != "private" else True
        logger.info(f"Pushing dataset '{dataset_name}' to Hugging Face Hub (privacy={privacy})")
        try:
            dataset.push_to_hub(config["configurations"]["hf_organization"] + "/" + dataset_name, private=privacy)
            logger.success(f"Successfully pushed dataset to Hugging Face Hub: {dataset_name}")
        except Exception as error:
            logger.error(f"Failed to push dataset to Hugging Face Hub: {str(error)}")
            raise
    else:
        logger.info(f"Saving dataset locally to: {dataset_name}")
        dataset.save_to_disk(dataset_name)
        logger.success(f"Successfully saved dataset to disk: {dataset_name}")


def create_multihop_chunks(config: dict):
    logger.info("Starting multihop chunks creation process")

    # load the dataset
    source_dataset_name = get_full_dataset_name(config)
    logger.debug(f"Loading source dataset: {source_dataset_name}")
    dataset = load_dataset(source_dataset_name, split="train")
    logger.info(f"Loaded source dataset with {len(dataset)} entries")

    # generate the multihop pairings
    logger.debug("Starting multihop pairings generation")
    multihop_pairings = generate_multihop_pairings(dataset)

    # save the multihop pairings
    output_dataset_name = config["selected_choices"]["make_multihop_chunks"]["multihop_pairings_dataset_name"]
    logger.debug(f"Preparing to save multihop pairings dataset: {output_dataset_name}")
    handle_dataset_push(multihop_pairings, output_dataset_name, config)

    logger.success("Multihop chunks creation completed successfully")
    return
