
import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasets import load_dataset, Dataset
from loguru import logger

# Configure loguru
logger.add(
    "logs/chunk_hf_data_{time}.log",
    rotation="500 MB",
    level="INFO",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}"
)

# Download NLTK data
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

def create_chunks_for_documents(document_with_summary_dataset_name: str, config: dict):
    """Create chunks for documents from a Hugging Face dataset.
    
    Args:
        document_with_summary_dataset_name (str): Name of the source dataset on Hugging Face
        config (dict): Configuration containing:
            - chunked_documents_dataset_name (str): Name for the output dataset
            - model_name (str): Name of the sentence transformer model
            - similarity_threshold (float): Threshold for semantic similarity
            - min_tokens (int): Minimum tokens per chunk
            - max_tokens (int): Maximum tokens per chunk
            - target_chunk_size (int): Target number of tokens per chunk
            - text_column (str): Name of the column containing the text to chunk
            - title_column (str, optional): Name of the column containing the title
    """
    logger.info(f"Loading dataset: {document_with_summary_dataset_name}")
    dataset = load_dataset(document_with_summary_dataset_name)

    # Initialize model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(config['model_name']).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        f"sentence-transformers/{config['model_name']}",
        use_fast=True,
        model_max_length=512
    )

    def process_document(example):
        # Get document text
        document_text = example[config['text_column']]
        
        # Get title if specified
        title = example.get(config.get('title_column', ''), '')
        
        # Create chunks
        chunks = semantic_chunking(
            document_text,
            model=model,
            tokenizer=tokenizer,
            similarity_threshold=config['similarity_threshold'],
            min_tokens=config['min_tokens'],
            max_tokens=config['max_tokens'],
            target_chunk_size=config['target_chunk_size']
        )
        
        return {
            'title': title,
            'chunks': chunks,
            'num_chunks': len(chunks)
        }

    # Process all documents
    logger.info("Processing documents...")
    chunked_dataset = dataset.map(
        process_document,
        remove_columns=dataset.column_names,
        desc="Creating chunks"
    )

    # Push to Hub
    logger.info(f"Pushing chunked dataset to: {config['chunked_documents_dataset_name']}")
    chunked_dataset.push_to_hub(config['chunked_documents_dataset_name'])
    
    logger.success("Chunking completed successfully")
    return chunked_dataset
