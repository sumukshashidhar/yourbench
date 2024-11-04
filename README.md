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
