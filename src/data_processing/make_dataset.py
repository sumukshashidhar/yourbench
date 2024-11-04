import os
import json
import uuid
from typing import List
from loguru import logger
from datasets import Dataset
import argparse

def read_chunks(
    chunk_directory: str,
    summary_directory: str,
) -> List[dict]:
    """Read and process document chunks from the specified directories."""
    document_paths = []
    for root, _, files in os.walk(chunk_directory):
        for file in files:
            document_paths.append(os.path.join(root, file))

    documents = []
    for document_path in document_paths:
        with open(document_path, "r") as file:
            chunk_data = json.load(file)

        # Construct related file paths
        summary_path = document_path.replace("semantic_chunks", "summaries").replace(".json", ".md")
        raw_path = document_path.replace("semantic_chunks", "raw").replace(".json", ".md")

        # Read summary if available
        summary_content = None
        if os.path.exists(summary_path):
            with open(summary_path, "r") as file:
                summary_content = file.read().strip()

        # Read raw document content
        with open(raw_path, "r") as file:
            raw_content = file.read().strip()

        # Extract document metadata
        document_type = document_path.split("/")[-2]
        document_name = document_path.split("/")[-1].replace(".json", "")
        document_id = str(uuid.uuid4())

        # Process each chunk
        for chunk_index, chunk_content in enumerate(chunk_data["chunks"]):
            documents.append({
                "title": chunk_data["title"].replace("#", ""),
                "chunk_uuid": str(uuid.uuid4()),
                "chunk_location_id": chunk_index,
                "chunk_size": len(chunk_content),
                "chunk": chunk_content,
                "summary": summary_content,
                "document_type": document_type,
                "document_name": document_name,
                "unique_document_id": document_id,
                "source_text": raw_content,
            })

    return documents

def make_huggingface_dataset(documents: List[dict]) -> Dataset:
    """Convert documents list to HuggingFace dataset."""
    return Dataset.from_list(documents)

def push_to_hub(dataset: Dataset, dataset_name: str, organization: str, private: bool = True):
    """Push dataset to HuggingFace Hub."""
    dataset.push_to_hub(f"{organization}/{dataset_name}", private=private)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process chunks into HuggingFace dataset')
    
    parser.add_argument('--chunk_directory', type=str, 
                       default="data/yourbench_y1/semantic_chunks",
                       help='Directory containing semantic chunks')
    
    parser.add_argument('--summary_directory', type=str,
                       default="data/yourbench_y1/summaries",
                       help='Directory containing summaries')
    
    parser.add_argument('--dataset_name', type=str,
                       default="y1",
                       help='Name for the HuggingFace dataset')
    
    parser.add_argument('--organization', type=str,
                       default="sumuks",
                       help='HuggingFace organization name')
    
    parser.add_argument('--private', type=bool,
                       default=True,
                       help='Whether to make the dataset private')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    logger.info("Starting dataset creation process")
    logger.info(f"Reading chunks from: {args.chunk_directory}")
    
    try:
        documents = read_chunks(args.chunk_directory, args.summary_directory)
        dataset = make_huggingface_dataset(documents)
        
        logger.info(f"Created dataset with {len(dataset)} chunks")
        
        push_to_hub(dataset, args.dataset_name, args.organization, args.private)
        
        logger.success(f"Dataset pushed to HuggingFace Hub: {args.organization}/{args.dataset_name}")
        logger.info(f"Total number of expanded chunks: {len(dataset)}")
        
    except Exception as error:
        logger.error(f"An error occurred during processing: {str(error)}")
        logger.exception("Full traceback:")
        raise

if __name__ == "__main__":
    main()