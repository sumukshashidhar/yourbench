import re
from typing import Dict

import nltk
import torch
from datasets import Dataset, load_dataset
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, GPT2Tokenizer


# Download NLTK data
nltk.download('punkt')


def _clean_text(text: str):
    """Clean text"""
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize quotes
    text = re.sub(r'[''"]', '"', text)
    # Normalize newlines
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def _create_chunk_boundaries(sentences: list, sentence_embeddings, chunking_configuration: Dict) -> list:
    """Calculate chunk boundaries based on semantic similarity"""
    similarities = [
        cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i + 1]])[0][0]
        for i in range(len(sentence_embeddings) - 1)
    ]

    boundaries = [0]
    for i, sim in enumerate(similarities):
        if sim < chunking_configuration["similarity_threshold"]:
            boundaries.append(i + 1)
    boundaries.append(len(sentences))
    return boundaries


def _process_segment(sentences: list, start: int, end: int, tokenizer, chunking_configuration: Dict,
                    current_chunk_sentences: list, current_token_count: int) -> tuple:
    """Process a segment of sentences and update chunks, including overlap"""
    segment_sentences = sentences[start:end]
    segment_text = ' '.join(segment_sentences)
    segment_tokens = tokenizer.tokenize(segment_text)
    segment_num_tokens = len(segment_tokens)
    chunks = []
    overlap_size = chunking_configuration.get("overlap_size", 2)  # Number of sentences to overlap

    if current_token_count + segment_num_tokens <= chunking_configuration["max_tokens"]:
        current_chunk_sentences.extend(segment_sentences)
        current_token_count += segment_num_tokens
        if current_token_count >= chunking_configuration["target_chunk_size"]:
            chunks.append(' '.join(current_chunk_sentences))
            # Keep last n sentences for overlap
            current_chunk_sentences = current_chunk_sentences[-overlap_size:] if overlap_size > 0 else []
            current_token_count = len(tokenizer.tokenize(' '.join(current_chunk_sentences)))
    else:
        if current_token_count >= chunking_configuration["min_tokens"]:
            chunks.append(' '.join(current_chunk_sentences))
            # Keep last n sentences for overlap
            overlap_sentences = current_chunk_sentences[-overlap_size:] if overlap_size > 0 else []
            current_chunk_sentences = overlap_sentences + segment_sentences
            current_token_count = len(tokenizer.tokenize(' '.join(current_chunk_sentences)))
        else:
            current_chunk_sentences.extend(segment_sentences)
            current_token_count += segment_num_tokens
            if current_token_count >= chunking_configuration["min_tokens"] or \
               current_token_count >= chunking_configuration["max_tokens"]:
                chunks.append(' '.join(current_chunk_sentences))
                # Keep last n sentences for overlap
                current_chunk_sentences = current_chunk_sentences[-overlap_size:] if overlap_size > 0 else []
                current_token_count = len(tokenizer.tokenize(' '.join(current_chunk_sentences)))

    return chunks, current_chunk_sentences, current_token_count


def semantic_chunking(document_text: str, chunking_configuration: Dict, model: SentenceTransformer, tokenizer: AutoTokenizer):
    """Semantic chunking of a document"""
    document_text = _clean_text(document_text)
    sentences = [s.strip() for s in sent_tokenize(document_text) if s.strip()]

    if not sentences:
        return []

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True).cpu().numpy()
    chunk_boundaries = _create_chunk_boundaries(sentences, sentence_embeddings, chunking_configuration)

    chunks = []
    current_chunk_sentences = []
    current_token_count = 0

    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        new_chunks, current_chunk_sentences, current_token_count = _process_segment(
            sentences, start, end, tokenizer, chunking_configuration,
            current_chunk_sentences, current_token_count
        )
        chunks.extend(new_chunks)

    if current_chunk_sentences:
        if current_token_count >= chunking_configuration["min_tokens"]:
            chunks.append(' '.join(current_chunk_sentences))
        else:
            if chunks:
                chunks[-1] += ' ' + ' '.join(current_chunk_sentences)
            else:
                chunks.append(' '.join(current_chunk_sentences))

    return chunks


def create_chunks_for_documents(hf_dataset_name: str, config: Dict):
    """Create chunks for documents"""
    # extract the chunking configuration
    chunking_configuration = config["chunking_configuration"]
    # check if we have a GPU + we're allowed to use it
    device = "cuda" if chunking_configuration["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    # load the model
    model = SentenceTransformer(chunking_configuration["model_name"], device=device)
    tokenizer = AutoTokenizer.from_pretrained(chunking_configuration["model_name"], use_fast=True, model_max_length=512)

    # load the dataset
    dataset = load_dataset(hf_dataset_name, split="train")

    # Add GPT2 tokenizer initialization
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')

    chunk_list = []
    for document in dataset:
        chunks = semantic_chunking(document["content"], chunking_configuration, model, tokenizer)
        rich_chunks = [{
            "id": document["id"],
            "title": document["title"],
            "summary": document["summary"],
            "chunk": chunk,
            "chunk_length": len(gpt2_tokenizer.encode(chunk))  # Add chunk length calculation
        } for chunk in chunks]
        chunk_list.extend(rich_chunks)

    chunks_dataset = Dataset.from_list(chunk_list)
    chunks_dataset.push_to_hub(config["datasets"]["chunked_doucments_dataset_name"], private=True)
