import random
from collections import defaultdict

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger

from utils.dataset_engine import handle_dataset_push, make_dataset_name


def generate_multihop_pairings(
    dataset: DatasetDict, pairing_configuration: dict
) -> Dataset:
    logger.info("Starting multihop pairings generation process")

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    logger.debug("Set random seeds for reproducibility")

    # Group by unique_document_id
    doc_groups = defaultdict(list)
    for idx, example in enumerate(dataset):
        doc_groups[example["document_id"]].append(
            {
                "chunk": example["chunk"],
                "chunk_location_id": example["chunk_location_id"],
                "document_name": example["document_name"],
                "document_summary": example["document_summary"],
                "document_category": example["document_category"],
            }
        )
    logger.info(f"Grouped {len(doc_groups)} unique documents for pairing generation")

    # Generate pairings
    pairings = []
    logger.debug("Starting to generate chunk combinations")

    for doc_id, chunks in doc_groups.items():
        if len(chunks) < pairing_configuration["min_num_chunks"]:
            logger.debug(
                f"Skipping document {doc_id} - insufficient chunks ({len(chunks)})"
            )
            continue

        n_chunks = len(chunks)
        n_combinations = max(1, int(np.sqrt(n_chunks)))
        logger.debug(
            f"Generating {n_combinations} combinations for document {doc_id} with {n_chunks} chunks"
        )

        # Calculate number of combinations to generate based on number of chunks
        n_combinations = max(
            1, int(np.sqrt(n_chunks))
        )  # Using square root as a heuristic

        # Generate combinations for different numbers of chunks (2-5)
        for _ in range(n_combinations):
            # Randomly choose number of chunks (2-5)
            num_chunks = random.randint(
                pairing_configuration["min_num_chunks"],
                min(
                    pairing_configuration["max_num_chunks"],
                    n_chunks,
                ),
            )

            # Randomly select chunks
            selected_chunks = random.sample(chunks, num_chunks)

            # Sort by chunk_location_id to maintain document order
            selected_chunks.sort(key=lambda x: x["chunk_location_id"])

            # Create new example
            new_example = {
                "document_id": doc_id,
                "document_name": selected_chunks[0]["document_name"],
                "document_summary": selected_chunks[0]["document_summary"],
                "document_category": selected_chunks[0]["document_category"],
                "chunk_ids": [chunk["chunk_location_id"] for chunk in selected_chunks],
                "num_chunks": num_chunks,
                "chunks": [chunk["chunk"] for chunk in selected_chunks],
            }

            pairings.append(new_example)

    logger.info(f"Generated {len(pairings)} multihop pairings in total")

    # Create new dataset
    logger.debug("Converting pairings to Hugging Face Dataset format")
    new_dataset = Dataset.from_dict(
        {
            "document_id": [p["document_id"] for p in pairings],
            "document_name": [p["document_name"] for p in pairings],
            "document_summary": [p["document_summary"] for p in pairings],
            "document_category": [p["document_category"] for p in pairings],
            "chunk_ids": [p["chunk_ids"] for p in pairings],
            "num_chunks": [p["num_chunks"] for p in pairings],
            "chunks": [p["chunks"] for p in pairings],
        }
    )
    logger.success(
        f"Successfully created dataset with {len(new_dataset)} multihop entries"
    )

    return new_dataset


def create_multihop_chunks(config: dict):
    logger.info("Starting multihop chunks creation process")

    # load the dataset
    source_dataset_name_key = config["pipeline"]["make_chunk_pairings"][
        "source_dataset_name"
    ]
    target_dataset_name_key = config["pipeline"]["make_chunk_pairings"][
        "target_dataset_name"
    ]
    pairing_configuration = config["pipeline"]["make_chunk_pairings"][
        "pairing_configuration"
    ]
    source_dataset_name = make_dataset_name(config, source_dataset_name_key)
    logger.debug(f"Loading source dataset: {source_dataset_name}")
    dataset = load_dataset(source_dataset_name, split="train")
    logger.info(f"Loaded source dataset with {len(dataset)} entries")

    # generate the multihop pairings
    logger.debug("Starting multihop pairings generation")
    multihop_pairings = generate_multihop_pairings(dataset, pairing_configuration)

    logger.debug(
        f"Preparing to save multihop pairings dataset: {target_dataset_name_key}"
    )
    handle_dataset_push(config, target_dataset_name_key, multihop_pairings)

    logger.success("Multihop chunks creation completed successfully")
    return
