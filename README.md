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
