import torch
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import json
import os
from collections import defaultdict
import multiprocessing as mp
import matplotlib.gridspec as gridspec
import argparse

# Add at the top of the file, after imports
mp.set_start_method('spawn', force=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Deduplicate questions in a dataset using semantic similarity')
    
    # Dataset arguments
    parser.add_argument('--input_dataset', type=str, default="sumuks/y1-questions-single-shot",
                       help='Input dataset name on HuggingFace Hub')
    parser.add_argument('--input_split', type=str, default="train",
                       help='Dataset split to process')
    parser.add_argument('--output_dataset', type=str, default="sumuks/y1-questions-single-shot-diverse",
                       help='Output dataset name for HuggingFace Hub')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, 
                       default='sentence-transformers/msmarco-MiniLM-L-12-v3',
                       help='SentenceTransformer model to use')
    
    # Processing arguments
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='Number of GPUs to use for processing')
    parser.add_argument('--similarity_threshold', type=float, default=0.9,
                       help='Threshold for considering questions similar')
    parser.add_argument('--batch_size', type=int, default=8192,
                       help='Batch size for similarity computation')
    
    # Output arguments
    parser.add_argument('--plots_dir', type=str, default='plots',
                       help='Directory to save plots')
    
    return parser.parse_args()

def plot_similarity_histogram(
    similarities,
    output_file='plots/similarity_histogram.png',
    title='b1-mini Question Similarity',
    x_label='Cosine Embedding Similarity',
    y_label='Frequency',
    bins=25,
    color='viridis',
    figsize=(5, 4),
    dpi=300
):
    """
    Plot a histogram of similarities with improved aesthetics.

    Args:
    similarities (array-like): The similarity data to plot.
    output_file (str): Path to save the output file.
    title (str): Title of the plot.
    x_label (str): Label for x-axis.
    y_label (str): Label for y-axis.
    bins (int): Number of bins for the histogram.
    color (str): Color map for the histogram.
    figsize (tuple): Figure size (width, height) in inches.
    dpi (int): Dots per inch for the output figure.
    """
    # Set up the figure and grid
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    # Plot the histogram
    n, bins, patches = ax.hist(similarities, bins=bins, edgecolor='none')

    # Color the bars using a colormap
    sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=min(similarities), vmax=max(similarities)))
    for bin, patch in zip(bins, patches):
        color = sm.to_rgba(bin)
        patch.set_facecolor(color)

    # Customize the plot
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(x_label, fontsize=14, labelpad=10)
    # ax.set_ylabel(y_label, fontsize=14, labelpad=10)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Increase font size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def output_high_similarity_clusters(clusters, dataset, output_base='plots/high_similarity_clusters'):
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    
    # Write to text file
    with open(f"{output_base}.txt", 'w', encoding='utf-8') as txt_file:
        txt_file.write("High Similarity Question Clusters\n")
        txt_file.write("=================================\n\n")
        
        for cluster_id, indices in tqdm(clusters.items(), desc="Writing clusters to text file"):
            txt_file.write(f"Cluster {cluster_id}:\n")
            txt_file.write("=" * 50 + "\n")
            
            for idx in indices:
                q_data = dataset[idx]
                txt_file.write(f"Document ID: {q_data['chunk_uuid']}\n")
                txt_file.write(f"Question: {q_data['question']}\n")
                txt_file.write(f"Answer: {q_data['answer']}\n")
                txt_file.write(f"Kind: {q_data['question_type']}\n")
                txt_file.write(f"Generating Model: {q_data['generator_model']}\n")
                txt_file.write("-" * 40 + "\n")
            
            txt_file.write("\n")
    
    # Write to JSONL file
    with open(f"{output_base}.jsonl", 'w', encoding='utf-8') as jsonl_file:
        for cluster_id, indices in tqdm(clusters.items(), desc="Writing clusters to JSONL"):
            cluster_data = {
                "cluster_id": cluster_id,
                "questions": []
            }
            for idx in indices:
                q_data = dataset[idx]
                cluster_data["questions"].append({
                    "document_id": q_data['chunk_uuid'],
                    "question": q_data['question'],
                    "answer": q_data['answer'],
                    "kind": q_data['question_type'],
                    "generating_model": q_data['generator_model']
                })
            jsonl_file.write(json.dumps(cluster_data) + '\n')

def compute_similarities_and_clusters(embeddings, threshold=0.9, batch_size=8192):
    device = embeddings.device
    n = embeddings.shape[0]
    similarities = []
    clusters = defaultdict(list)
    processed = set()
    
    # Normalize embeddings for faster cosine similarity computation
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    for i in tqdm(range(0, n, batch_size), desc="Computing similarities and clusters"):
        end = min(i + batch_size, n)
        batch = embeddings[i:end]
        
        # Compute cosine similarities for the batch
        sim_matrix = torch.mm(batch, embeddings.t())
        
        for j in range(batch.shape[0]):
            if i + j in processed:
                continue
            
            sim_row = sim_matrix[j]
            max_sim = torch.max(sim_row).item()
            similarities.append(max_sim)
            
            # Find indices of similar questions
            similar_indices = torch.where(sim_row >= threshold)[0].cpu().numpy()
            
            if len(similar_indices) > 1:  # Include the question itself
                cluster_id = len(clusters)
                clusters[cluster_id] = similar_indices.tolist()
                processed.update(similar_indices)
            
            processed.add(i + j)
    
    return similarities, clusters


def deduplicate_dataset(dataset, clusters):
    to_keep = set()
    for cluster in clusters.values():
        to_keep.add(random.choice(cluster))
    
    deduplicated_dataset = dataset.select([i for i in range(len(dataset)) if i in to_keep])
    return deduplicated_dataset, len(dataset) - len(deduplicated_dataset)

def process_on_gpu(args):
    chunk_dataset, gpu_id, model_name = args
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    model = SentenceTransformer(model_name).to(device)
    
    embeddings = model.encode(chunk_dataset['question'], show_progress_bar=True, device=device, convert_to_tensor=True)
    similarities, clusters = compute_similarities_and_clusters(embeddings)
    
    return similarities, clusters

def analyze_and_deduplicate_questions(dataset, args):
    torch.cuda.empty_cache()
    
    # Split dataset into chunks for each GPU
    chunk_size = len(dataset) // args.num_gpus
    chunks = [dataset.select(range(i * chunk_size, (i + 1) * chunk_size if i < args.num_gpus - 1 else len(dataset))) 
              for i in range(args.num_gpus)]
    
    # Prepare arguments for multiprocessing
    process_args = [(chunk, i, args.model_name) for i, chunk in enumerate(chunks)]
    
    # Use multiprocessing to run GPU tasks in parallel
    with mp.Pool(args.num_gpus) as pool:
        results = pool.map(process_on_gpu, process_args)
    
    # Combine results from all GPUs
    all_similarities = []
    all_clusters = defaultdict(list)
    cluster_offset = 0
    for i, (similarities, clusters) in enumerate(results):
        all_similarities.extend(similarities)
        for cluster_id, indices in clusters.items():
            all_clusters[cluster_id + cluster_offset] = [idx + i * chunk_size for idx in indices]
        cluster_offset += len(clusters)
    
    # Plot histogram
    plot_similarity_histogram(all_similarities, 
                            output_file=os.path.join(args.plots_dir, 'similarity_histogram.png'))
    
    # Output high-similarity clusters to file
    output_high_similarity_clusters(all_clusters, dataset, 
                                  output_base=os.path.join(args.plots_dir, 'high_similarity_clusters'))
    
    # Deduplicate dataset
    deduplicated_dataset, num_removed = deduplicate_dataset(dataset, all_clusters)
    
    return deduplicated_dataset, all_similarities, num_removed, all_clusters

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Load dataset
    dataset = load_dataset(args.input_dataset, split=args.input_split)
    
    try:
        original_dataset_size = len(dataset)
        deduplicated_dataset, similarities, num_removed, clusters = analyze_and_deduplicate_questions(dataset, args)
        
        print(f"Original dataset size: {original_dataset_size}")
        print(f"Deduplicated dataset size: {len(deduplicated_dataset)}")
        print(f"Number of questions removed: {num_removed}")
        print(f"Number of clusters: {len(clusters)}")
        print(f"Mean similarity: {np.mean(similarities):.4f}")
        print(f"Median similarity: {np.median(similarities):.4f}")
        print(f"Max similarity: {np.max(similarities):.4f}")
        print(f"Min similarity: {np.min(similarities):.4f}")
        
        if args.output_dataset:
            deduplicated_dataset.push_to_hub(args.output_dataset)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()