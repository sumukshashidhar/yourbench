# YourBench Quick Start Guide

## 🚀 Fast Mode - Generate Datasets in One Command

YourBench now supports a quick mode that allows you to generate evaluation datasets with a single command:

```bash
yourbench run --model gpt-4o --docs path/to/your/documents
```

This will:
- Process your documents (PDFs, text files, HTML, etc.)
- Generate high-quality Q&A pairs using the specified model
- Export results as JSONL files in your current directory

## Examples

### 1. Basic Usage - Generate Questions from a PDF

```bash
yourbench run --model gpt-4o --docs research_paper.pdf
```

Output files created:
- `dataset.jsonl` - All generated Q&A pairs
- `train.jsonl`, `test.jsonl` - If dataset is split

### 2. Generate from Multiple Documents

```bash
yourbench run --model gpt-4o --docs ./documents/
```

Processes all supported files in the directory.

### 3. Push to HuggingFace Hub

```bash
yourbench run --model gpt-4o --docs paper.pdf --push-to-hub myusername/my-qa-dataset
```

This will:
- Generate the dataset
- Save JSONL files locally
- Upload to HuggingFace Hub as a private dataset

### 4. Using Different Models

```bash
# OpenAI Models
yourbench run --model gpt-3.5-turbo --docs docs/

# Open Models via HuggingFace
yourbench run --model meta-llama/Llama-3-8B-Instruct --docs docs/

# Local Models (requires setup)
yourbench run --model llama3:8b --docs docs/
```

## Environment Setup

Before using quick mode, set up your API keys:

```bash
# For OpenAI models
export OPENAI_API_KEY="your-key-here"

# For HuggingFace Hub upload
export HF_TOKEN="your-hf-token"
export HF_ORGANIZATION="your-username"  # Optional
```

## Output Format

The generated JSONL files contain questions and answers in this format:

```json
{"question": "What is the main contribution of this paper?", "answer": "The paper introduces...", "context": "..."}
{"question": "How does the proposed method work?", "answer": "The method works by...", "context": "..."}
```

## Advanced Usage

For more control over the pipeline, create a configuration file:

```bash
# Interactive config creator
yourbench create

# Run with custom config
yourbench run config.yaml
```

## Performance

- **Startup time**: ~0.07s (4x faster than v0.3)
- **JSONL export**: Instant, no conversion needed
- **Minimal dependencies**: Core features work without heavy packages

## Tips

1. **Model Selection**: GPT-4 variants produce higher quality questions but cost more. GPT-3.5-turbo offers good balance.

2. **Document Preparation**: 
   - PDFs work best when text-based (not scanned images)
   - Break very large documents into chapters for better results

3. **Batch Processing**: Process multiple related documents together for cross-document questions:
   ```bash
   yourbench run --model gpt-4o --docs ./research_papers/
   ```

4. **Local Models**: For privacy-sensitive documents, use local models:
   ```bash
   yourbench run --model llama3:70b --docs confidential.pdf
   ```

## Troubleshooting

- **"No API key found"**: Set environment variables (see Environment Setup)
- **"Model not found"**: Check model name spelling and availability
- **Empty output**: Ensure documents contain extractable text
- **Slow processing**: Large documents take time; use `--debug` to see progress

For detailed documentation, visit the [YourBench GitHub repository](https://github.com/huggingface/yourbench).