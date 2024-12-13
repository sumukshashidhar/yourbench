# Document Chunking in Yourbench

This document explains how Yourbench performs semantic chunking of documents for question generation.

## Overview

Yourbench uses a sophisticated semantic chunking approach that:
1. Maintains semantic coherence
2. Respects token limits
3. Creates overlapping chunks when needed
4. Uses sentence transformers for similarity

## Chunking Process

### 1. Text Preprocessing

```python
def _clean_text(text: str):
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize quotes
    text = re.sub(r'[''"]', '"', text)
    # Normalize newlines
    text = re.sub(r'\n+', '\n', text)
    return text.strip()
```

- Normalizes whitespace and quotes
- Removes redundant newlines
- Ensures consistent text format

### 2. Sentence Splitting

The document is split into sentences using NLTK's sentence tokenizer:
```python
sentences = sent_tokenize(document_text)
```

### 3. Semantic Boundary Detection

```python
def _create_chunk_boundaries(sentences, sentence_embeddings, config):
    # Calculate similarities between adjacent sentences
    similarities = [
        cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        for i in range(len(embeddings) - 1)
    ]
    
    # Create boundaries where similarity is below threshold
    boundaries = [0]
    for i, sim in enumerate(similarities):
        if sim < config["similarity_threshold"]:
            boundaries.append(i + 1)
    boundaries.append(len(sentences))
    return boundaries
```

- Uses sentence transformers for embeddings
- Calculates cosine similarity between adjacent sentences
- Creates boundaries at semantic shifts
- Configurable similarity threshold

### 4. Chunk Size Control

The chunking process respects three size parameters:

1. `min_tokens`: Minimum chunk size
2. `target_chunk_size`: Desired chunk size
3. `max_tokens`: Maximum chunk size

```yaml
chunking_configuration:
    min_tokens: 256
    target_chunk_size: 512
    max_tokens: 1024
```

### 5. Chunk Processing

For each semantic segment:

1. Calculate token count
2. If within limits:
   - Add to current chunk
   - Check against target size
3. If over limits:
   - Create new chunk
   - Maintain overlap

## Overlap Handling

The system maintains overlap between chunks to preserve context:

```python
# Keep last n sentences for overlap
overlap_sentences = current_chunk_sentences[-overlap_size:]
current_chunk_sentences = overlap_sentences + segment_sentences
```

- Default overlap: 2 sentences
- Helps maintain context across chunks
- Prevents information loss at boundaries

## Configuration Parameters

### Model Configuration
```yaml
chunking_configuration:
    model_name: sentence-transformers/all-MiniLM-L6-v2
    device: cuda  # or cpu
```

- Uses sentence transformers for embeddings
- Supports GPU acceleration
- Configurable model selection

### Size Parameters
```yaml
chunking_configuration:
    min_tokens: 256
    target_chunk_size: 512
    max_tokens: 1024
```

- Controls chunk size bounds
- Based on token counts
- Flexible configuration

### Similarity Control
```yaml
chunking_configuration:
    similarity_threshold: 0.3
```

- Controls semantic boundary detection
- Lower values: more boundaries
- Higher values: fewer boundaries

## Output Format

Each chunk includes metadata:

```python
{
    "id": document_id,
    "title": document_title,
    "summary": document_summary,
    "chunk": chunk_text,
    "chunk_location_id": index,
    "chunk_length": token_count
}
```

## Best Practices

1. **Model Selection**
   - Use appropriate sentence transformer
   - Consider speed vs. accuracy
   - Match model to content type

2. **Size Configuration**
   - Balance chunk size with content
   - Consider model context limits
   - Adjust for content complexity

3. **Similarity Threshold**
   - Test with sample content
   - Adjust for content structure
   - Balance cohesion vs. size

4. **Device Selection**
   - Use GPU for large datasets
   - Consider available resources
   - Balance speed vs. cost

## Performance Considerations

1. **GPU Acceleration**
   - Significantly faster embedding
   - Required for large datasets
   - Configurable via `device` parameter

2. **Batch Processing**
   - Processes documents in parallel
   - Efficient resource usage
   - Automatic batching of embeddings

3. **Memory Management**
   - Efficient tensor handling
   - Automatic garbage collection
   - Streaming for large documents

## Error Handling

The chunking system includes robust error handling:

1. Empty documents
2. Invalid sentences
3. Token count errors
4. Model loading issues

## Integration Points

The chunking system integrates with:

1. Dataset generation
2. Question generation
3. Multi-hop pairing
4. Hugging Face datasets