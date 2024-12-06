from typing import Dict
import torch
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, GPT2Tokenizer
from datasets import load_dataset, Dataset
import re

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

def semantic_chunking(document_text: str, chunking_configuration: Dict, model: SentenceTransformer, tokenizer: AutoTokenizer):
    """Semantic chunking of a document"""
    document_text = _clean_text(document_text)
    # extract sentences
    sentences = sent_tokenize(document_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Handle empty documents
    if not sentences:
        return []
    # encode sentences
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
        if sim < chunking_configuration["similarity_threshold"]:
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
        if current_token_count + segment_num_tokens <= chunking_configuration["max_tokens"]:
            current_chunk_sentences.extend(segment_sentences)
            current_token_count += segment_num_tokens
            # If current chunk reaches target size, finalize it
            if current_token_count >= chunking_configuration["target_chunk_size"]:
                chunks.append(' '.join(current_chunk_sentences))
                current_chunk_sentences = []
                current_token_count = 0
        else:
            # Check if current chunk meets min_tokens
            if current_token_count >= chunking_configuration["min_tokens"]:
                chunks.append(' '.join(current_chunk_sentences))
                current_chunk_sentences = segment_sentences
                current_token_count = segment_num_tokens
            else:
                # Attempt to adjust the chunk to meet min_tokens
                current_chunk_sentences.extend(segment_sentences)
                current_token_count += segment_num_tokens
                if current_token_count >= chunking_configuration["min_tokens"]:
                    chunks.append(' '.join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_token_count = 0
                elif current_token_count >= chunking_configuration["max_tokens"]:
                    # Finalize the chunk even if it doesn't meet min_tokens
                    chunks.append(' '.join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_token_count = 0

    # Add any remaining sentences to chunks
    if current_chunk_sentences:
        if current_token_count >= chunking_configuration["min_tokens"]:
            chunks.append(' '.join(current_chunk_sentences))
        else:
            if chunks:
                chunks[-1] += ' ' + ' '.join(current_chunk_sentences)
            else:
                chunks.append(' '.join(current_chunk_sentences))

    print(chunks)
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
    dataset = dataset.select(range(1))

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
    
    