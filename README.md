# yourbench
Benchmark Large Language Models Reliably On Your Data

# Usage
## Data Preparation

1. **Data Structure**
   - Reference the `data/yourbench_y1` to see how the data is structured.
   - Add your raw markdown files to `data/YOURDATASET/raw` and your summaries to `data/YOURDATASET/summaries`.
     - Note: We do not yet provide a script to generate summaries, but this is an area of active development.

2. **Semantic Chunking**
   - Process your raw documents into semantically meaningful chunks using the chunking script:
   ```bash
   python src/data_processing/chunk_raw_data.py \
     --input_directory data/YOURDATASET/raw \
     --output_directory data/YOURDATASET/semantic_chunks \
     --model_name all-mpnet-base-v2 \
     --similarity_threshold 0.9 \
     --min_tokens 256 \
     --max_tokens 1024 \
     --target_chunk_size 512
   ```

   Parameters:
   - `--input_directory`: Path to your raw documents (default: data/yourbench_y1/raw)
   - `--output_directory`: Where to save the chunked documents (default: data/yourbench_y1/semantic_chunks)
   - `--model_name`: Sentence transformer model for semantic analysis (default: all-mpnet-base-v2)
   - `--similarity_threshold`: Threshold for determining chunk boundaries (default: 0.9)
   - `--min_tokens`: Minimum tokens per chunk (default: 256)
   - `--max_tokens`: Maximum tokens per chunk (default: 1024)
   - `--target_chunk_size`: Ideal number of tokens per chunk (default: 512)

   The script will:
   - Process each document in the input directory
   - Create semantically coherent chunks based on the specified parameters
   - Save the chunks as JSON files in the output directory
   - Generate detailed logs in the `logs` directory

3. **Create HuggingFace Dataset**
   - Convert the processed chunks into a HuggingFace dataset and upload it to the Hub:
   ```bash
   python src/data_processing/make_dataset.py \
     --chunk_directory data/YOURDATASET/semantic_chunks \
     --summary_directory data/YOURDATASET/summaries \
     --dataset_name your-dataset-name \
     --organization your-org \
     --private true
   ```

   Parameters:
   - `--chunk_directory`: Path to your processed chunks (default: data/yourbench_y1/semantic_chunks)
   - `--summary_directory`: Path to your summaries (default: data/yourbench_y1/summaries)
   - `--dataset_name`: Name for your HuggingFace dataset (default: b1-mini-raw-gpt)
   - `--organization`: Your HuggingFace organization name (default: anon)
   - `--private`: Whether to make the dataset private (default: true)

   The script will:
   - Read all chunks and their corresponding summaries
   - Create a HuggingFace dataset with document metadata
   - Upload the dataset to the HuggingFace Hub
   - Generate logs of the process

4. **Create Multi-hop Pairings**
   - Generate multi-hop combinations from your dataset by combining 2-5 related chunks:
   ```bash
   python src/data_processing/make_multihop_pairings.py \
     --source_dataset your-org/your-dataset-name \
     --output_dataset your-org/your-dataset-multihop
   ```

   Parameters:
   - `--source_dataset`: Your source HuggingFace dataset ID
   - `--output_dataset`: Output dataset ID for the multi-hop version

   The script will:
   - Group chunks from the same document together
   - Generate random combinations of 2-5 sequential chunks
   - Create a new dataset with these multi-hop pairings
   - Number of combinations per document scales with document length (using sqrt)
   - Maintains original document order within each pairing
   - Preserves all relevant metadata (title, document type, etc.)
   - Upload the resulting dataset to HuggingFace Hub as private

## Question Generation

After preparing your dataset, you can generate questions using various question types:

```bash
python src/generation/generate_single_shot_questions.py \
  --dataset "your-org/your-dataset-name" \
  --split "train" \
  --output-dataset "your-org/your-dataset-questions" \
  --question-types analytical factual conceptual \
  --strategy "openai" \
  --api-key "your-api-key" \
  --base-url "http://your-inference-endpoint/v1/" \
  --model "your-model-name" \
  --max-concurrent 1024
```

Parameters:
- `--dataset`: HuggingFace dataset ID (default: "sumuks/y1")
- `--split`: Dataset split to use (default: "train")
- `--output-dataset`: Output dataset ID on HuggingFace (default: "sumuks/y1-questions-x2")
- `--question-types`: List of question types to generate. Available types:
  - analytical
  - application-based
  - clarification
  - conceptual
  - counterfactual
  - edge-case
  - factual
  - false-premise
  - open-ended
  - true-false
- `--strategy`: Inference strategy (default: "openai")
- `--api-key`: API key for inference (default: "EMPTY")
- `--base-url`: Base URL for inference (default: "http://localhost:30000/v1/")
- `--model`: Model name (default: "mistralai/Mistral-Small-Instruct-2409")
- `--max-concurrent`: Maximum concurrent requests (default: 1024)

The script will:
- Load your dataset from HuggingFace
- Generate questions for each specified question type
- Upload the generated questions to a new HuggingFace dataset
- Generate detailed logs of the generation process

Note: The question generation process requires access to an inference endpoint (OpenAI API or compatible).

### Multi-hop Question Generation

For generating questions that require reasoning across multiple chunks of text:

```bash
python src/generation/generate_multihop_questions_local.py \
  --dataset "your-org/your-dataset-multihop" \
  --split "train" \
  --output-dataset "your-org/your-dataset-multihop-questions" \
  --question-types analytical factual conceptual \
  --strategy "openai" \
  --api-key "your-api-key" \
  --base-url "http://your-inference-endpoint/v1/" \
  --model "your-model-name" \
  --max-concurrent 1024 \
  --start-server
```

Parameters:
- All parameters from single-shot generation, plus:
- `--start-server`: Optional flag to start a local vLLM OpenAI-compatible server

The script will:
- Start a local vLLM server if requested (requires GPU)
- Generate questions that require synthesizing information across multiple chunks
- Support all question types from single-shot generation
- Automatically handle server lifecycle and cleanup
- Generate detailed generation statistics in `logs/generation_stats_multihop.jsonl`

Server Configuration:
- Uses vLLM for efficient inference
- Runs on port 3000 by default
- Configured with tensor parallelism (4 GPUs)
- Enables prefix caching for better performance
- Auto-selects optimal dtype based on model and hardware

Note: The input dataset should be preprocessed using `make_multihop_pairings.py` as described in the Data Preparation section.
