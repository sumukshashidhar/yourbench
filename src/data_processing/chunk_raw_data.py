import os
import json
from pathlib import Path
import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import statistics
import csv
import argparse
from loguru import logger

# Configure loguru
logger.add(
    "logs/chunk_raw_data_{time}.log",
    rotation="500 MB",
    level="INFO",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}"
)

# Corrected NLTK download
nltk.download('punkt')

def semantic_chunking(
    document_text, model, tokenizer, similarity_threshold, min_tokens, max_tokens, target_chunk_size
):
    # Data Cleaning: Remove extra whitespace
    document_text = re.sub(r'\s+', ' ', document_text).strip()

    # Sentence Segmentation
    sentences = sent_tokenize(document_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Handle empty documents
    if not sentences:
        return []

    # Compute embeddings for sentences
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    sentence_embeddings = sentence_embeddings.cpu().numpy()

    # Calculate Semantic Similarity between adjacent sentences
    similarities = [
        cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i + 1]])[0][0]
        for i in range(len(sentence_embeddings) - 1)
    ]

    # Identify Chunk Boundaries based on similarity threshold
    chunk_boundaries = [0]
    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            chunk_boundaries.append(i + 1)
    chunk_boundaries.append(len(sentences))

    # Initialize variables
    chunks = []
    current_chunk_sentences = []
    current_token_count = 0
    idx = 0

    # Iterate over chunk boundaries
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        segment_sentences = sentences[start:end]
        segment_text = ' '.join(segment_sentences)
        segment_tokens = tokenizer.tokenize(segment_text)
        segment_num_tokens = len(segment_tokens)

        # Try to build chunks aiming for the target_chunk_size
        if current_token_count + segment_num_tokens <= max_tokens:
            current_chunk_sentences.extend(segment_sentences)
            current_token_count += segment_num_tokens
            # If current chunk reaches target size, finalize it
            if current_token_count >= target_chunk_size:
                chunks.append(' '.join(current_chunk_sentences))
                current_chunk_sentences = []
                current_token_count = 0
        else:
            # Check if current chunk meets min_tokens
            if current_token_count >= min_tokens:
                chunks.append(' '.join(current_chunk_sentences))
                current_chunk_sentences = segment_sentences
                current_token_count = segment_num_tokens
            else:
                # Attempt to adjust the chunk to meet min_tokens
                current_chunk_sentences.extend(segment_sentences)
                current_token_count += segment_num_tokens
                if current_token_count >= min_tokens:
                    chunks.append(' '.join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_token_count = 0
                elif current_token_count >= max_tokens:
                    # Finalize the chunk even if it doesn't meet min_tokens
                    chunks.append(' '.join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_token_count = 0

    # Add any remaining sentences to chunks
    if current_chunk_sentences:
        if current_token_count >= min_tokens:
            chunks.append(' '.join(current_chunk_sentences))
        else:
            if chunks:
                chunks[-1] += ' ' + ' '.join(current_chunk_sentences)
            else:
                chunks.append(' '.join(current_chunk_sentences))

    return chunks

def process_files_with_settings(input_dir, output_dir, settings_list, model_name):
    # Update model initialization to use model_name parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        f"sentence-transformers/{model_name}",
        use_fast=True,
        model_max_length=512
    )

    # Store the chunk lengths for each setting
    settings_chunk_lengths = []

    for settings in settings_list:
        similarity_threshold = settings['similarity_threshold']
        min_tokens = settings['min_tokens']
        max_tokens = settings['max_tokens']
        target_chunk_size = settings['target_chunk_size']

        logger.info(f"Processing with settings: {settings}")

        chunk_lengths = []

        for root, _, files in os.walk(input_dir):
            for file in files:
                input_path = Path(root) / file
                relative_path = input_path.relative_to(input_dir)
                output_path = Path(output_dir) / relative_path.with_suffix(".json")
                
                logger.debug(f"Processing file: {input_path}")

                with open(input_path, "r", encoding="utf-8") as f:
                    document_text = f.read()

                # Extract the title (first non-empty line) and the rest of the content
                lines = document_text.strip().split("\n")
                title = ""
                content = ""

                for line in lines:
                    if line.strip():
                        title = line.strip()
                        break

                content = document_text.replace(title, '', 1).strip()

                # Ensure the title doesn't contain markdown headers
                title = re.sub(r'^#+\s*', '', title)

                chunks = semantic_chunking(
                    content,
                    model=model,
                    tokenizer=tokenizer,
                    similarity_threshold=similarity_threshold,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    target_chunk_size=target_chunk_size
                )

                # Collect chunk lengths
                for chunk in chunks:
                    tokens = tokenizer.tokenize(chunk)
                    chunk_lengths.append(len(tokens))

                logger.debug(f"Generated {len(chunks)} chunks for {input_path}")

                # Write chunks to file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"title": title, "chunks": chunks},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                logger.debug(f"Wrote chunks to {output_path}")

        # Compute statistics on chunk lengths
        mean_length = statistics.mean(chunk_lengths) if chunk_lengths else 0
        stdev_length = statistics.stdev(chunk_lengths) if len(chunk_lengths) > 1 else 0
        variance_length = statistics.variance(chunk_lengths) if len(chunk_lengths) > 1 else 0

        settings_chunk_lengths.append({
            'settings': settings,
            'chunk_lengths': chunk_lengths,
            'mean': mean_length,
            'stdev': stdev_length,
            'variance': variance_length
        })

    return settings_chunk_lengths

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process documents into semantic chunks')
    
    # Model settings
    parser.add_argument('--model_name', type=str, default='all-mpnet-base-v2',
                      help='Name of the sentence transformer model')
    
    # Directory paths
    parser.add_argument('--input_directory', type=str, default='data/yourbench_y1/raw',
                      help='Input directory containing raw documents')
    parser.add_argument('--output_directory', type=str, default='data/yourbench_y1/semantic_chunks',
                      help='Output directory for chunked documents')
    
    # Chunking parameters
    parser.add_argument('--similarity_threshold', type=float, default=0.9,
                      help='Similarity threshold for chunk boundaries')
    parser.add_argument('--min_tokens', type=int, default=256,
                      help='Minimum number of tokens per chunk')
    parser.add_argument('--max_tokens', type=int, default=1024,
                      help='Maximum number of tokens per chunk')
    parser.add_argument('--target_chunk_size', type=int, default=512,
                      help='Target number of tokens per chunk')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("Starting document chunking process")
    logger.info(f"Using model: {args.model_name}")
    logger.info(f"Input directory: {args.input_directory}")
    logger.info(f"Output directory: {args.output_directory}")
    
    # Create settings list using command line arguments
    settings_list = [
        {
            'similarity_threshold': args.similarity_threshold,
            'min_tokens': args.min_tokens,
            'max_tokens': args.max_tokens,
            'target_chunk_size': args.target_chunk_size
        }
    ]

    try:
        # Process files with parsed arguments
        settings_chunk_lengths = process_files_with_settings(
            args.input_directory,
            args.output_directory,
            settings_list,
            args.model_name
        )

        # Analyze and log results
        for result in settings_chunk_lengths:
            settings = result['settings']
            mean_length = result['mean']
            stdev_length = result['stdev']
            variance_length = result['variance']
            chunk_count = len(result['chunk_lengths'])
            
            logger.info("Chunking Results:")
            logger.info(f"Settings: {settings}")
            logger.info(f"Mean chunk length: {mean_length:.2f} tokens")
            logger.info(f"Standard deviation of chunk lengths: {stdev_length:.2f}")
            logger.info(f"Variance of chunk lengths: {variance_length:.2f}")
            logger.info(f"Number of chunks: {chunk_count}")
            logger.info("-----")

        logger.success("Document chunking completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        logger.exception("Full traceback:")
        raise

if __name__ == "__main__":
    main()