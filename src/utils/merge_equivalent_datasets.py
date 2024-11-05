from datasets import load_dataset, concatenate_datasets
from huggingface_hub import HfApi
import argparse
import logging
import os
from dotenv import load_dotenv

load_dotenv()

def merge_datasets(input_datasets: list[str], output_dataset: str) -> None:
    """Merge multiple datasets into one and push to HuggingFace"""
    datasets = []
    
    for dataset_id in input_datasets:
        logging.info(f"Loading dataset: {dataset_id}")
        try:
            dataset = load_dataset(dataset_id, split='train')
            datasets.append(dataset)
        except Exception as e:
            logging.error(f"Failed to load {dataset_id}: {e}")
            raise

    logging.info("Concatenating datasets...")
    merged_dataset = concatenate_datasets(datasets)
    logging.info(f"Final dataset size: {len(merged_dataset)} entries")

    logging.info(f"Pushing merged dataset to {output_dataset}")
    merged_dataset.push_to_hub(output_dataset, private=True)

def main():
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Merge multiple HuggingFace datasets into one')
    parser.add_argument('input_datasets', nargs='+', help='List of input dataset IDs (username/dataset-name)')
    parser.add_argument('output_dataset', help='Output dataset ID (username/dataset-name)')
    
    args = parser.parse_args()
    merge_datasets(args.input_datasets, args.output_dataset)

if __name__ == "__main__":
    main()
