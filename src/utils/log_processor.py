from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import HfApi
import json
import os
from datetime import datetime
import logging
from typing import List, Dict, Any
import shutil
from dotenv import load_dotenv

load_dotenv()

def read_log_file(log_path: str) -> List[Dict[str, Any]]:
    """
    Read and parse the log file, returning a list of log entries.
    """
    entries = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                # Split the line by ' | ' since logs are in format: "timestamp | level | json_content"
                parts = line.strip().split(' | ')
                if len(parts) >= 3:
                    # Parse the JSON content (third part)
                    log_entry = json.loads(parts[2])
                    entries.append(log_entry)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse log line: {line.strip()}")
                continue
            except Exception as e:
                logging.error(f"Error processing line: {line.strip()}, Error: {str(e)}")
                continue
    return entries

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten nested dictionaries for dataset compatibility.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def process_entries(entries: List[Dict[str, Any]]) -> Dict[str, List]:
    """
    Process log entries into a format suitable for HuggingFace datasets.
    """
    dataset_dict = {
        'strategy': [],
        'model': [],
        'temperature': [],
        'max_tokens': [],
        'messages': [],
        'output': [],
        'start_time': [],
        'end_time': [],
        'time_taken_seconds': [],
        'status': [],
        'prompt_tokens': [],
        'completion_tokens': [],
        'total_tokens': [],
        'error': []
    }
    
    for entry in entries:
        # Flatten hyperparameters
        hyperparams = entry.get('hyperparameters', {})
        
        # Add base fields
        dataset_dict['strategy'].append(entry.get('strategy', ''))
        dataset_dict['model'].append(entry.get('model', ''))
        dataset_dict['temperature'].append(hyperparams.get('temperature', 0.0))
        dataset_dict['max_tokens'].append(hyperparams.get('max_tokens', 0))
        dataset_dict['messages'].append(json.dumps(entry.get('messages', [])))
        dataset_dict['output'].append(entry.get('output', ''))
        dataset_dict['start_time'].append(entry.get('start_time', ''))
        dataset_dict['end_time'].append(entry.get('end_time', ''))
        dataset_dict['time_taken_seconds'].append(entry.get('time_taken_seconds', 0))
        dataset_dict['status'].append(entry.get('status', ''))
        
        # Process usage information
        usage = entry.get('usage', {})
        dataset_dict['prompt_tokens'].append(usage.get('prompt_tokens', 0))
        dataset_dict['completion_tokens'].append(usage.get('completion_tokens', 0))
        dataset_dict['total_tokens'].append(usage.get('total_tokens', 0))
        
        # Add error information if present
        dataset_dict['error'].append(entry.get('error', ''))
    
    return dataset_dict

def backup_log_file(log_path: str):
    """
    Create a backup of the log file with timestamp.
    """
    if not os.path.exists(log_path):
        return
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = 'log_backups'
    os.makedirs(backup_dir, exist_ok=True)
    
    backup_path = os.path.join(backup_dir, f'inference_{timestamp}.log')
    shutil.copy2(log_path, backup_path)
    return backup_path

def clear_log_file(log_path: str):
    """
    Clear the contents of the log file while preserving the file.
    """
    open(log_path, 'w').close()

def dataset_exists(repo_id: str) -> bool:
    """
    Check if the dataset already exists on HuggingFace.
    """
    try:
        api = HfApi()
        api.dataset_info(repo_id)
        return True
    except Exception:
        return False

def load_existing_dataset(repo_id: str) -> Dataset:
    """
    Load the existing dataset from HuggingFace.
    """
    try:
        return load_dataset(repo_id, split='train')
    except Exception as e:
        logging.error(f"Error loading existing dataset: {str(e)}")
        raise

def push_to_huggingface(
    dataset_dict: Dict[str, List],
    repo_id: str,
) -> None:
    """
    Push the processed logs to HuggingFace as a dataset, concatenating with existing data if present.
    """
    # Convert new logs to Dataset
    new_dataset = Dataset.from_dict(dataset_dict)
    
    # Check if dataset already exists
    if dataset_exists(repo_id):
        logging.info("Found existing dataset. Loading and concatenating...")
        try:
            # Load existing dataset
            existing_dataset = load_existing_dataset(repo_id)
            
            # Concatenate datasets
            combined_dataset = concatenate_datasets([existing_dataset, new_dataset])
            logging.info(f"Combined dataset size: {len(combined_dataset)} entries")
        except Exception as e:
            logging.error(f"Error while concatenating datasets: {str(e)}")
            raise
    else:
        logging.info("No existing dataset found. Creating new dataset...")
        combined_dataset = new_dataset
    
    # Push to hub
    combined_dataset.push_to_hub(repo_id, private=True)

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    LOG_PATH = "logs/inference.log"
    REPO_ID = os.getenv("BETTERBENCH_INFERENCE_DATASET")  # format: "username/dataset-name"
    try:
        # Read log entries
        logging.info("Reading log entries...")
        entries = read_log_file(LOG_PATH)
        
        if not entries:
            logging.info("No new log entries to process")
            return
        
        # Process entries into dataset format
        logging.info("Processing entries...")
        dataset_dict = process_entries(entries)
        
        # Create backup before clearing
        logging.info("Creating backup...")
        backup_path = backup_log_file(LOG_PATH)
        logging.info(f"Backup created at: {backup_path}")
        
        # Push to HuggingFace (now with concatenation)
        logging.info("Pushing to HuggingFace...")
        push_to_huggingface(dataset_dict, REPO_ID)
        
        # Clear the log file
        logging.info("Clearing log file...")
        clear_log_file(LOG_PATH)
        
        logging.info("Processing completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()