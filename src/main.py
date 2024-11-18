import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict
from loguru import logger
from dotenv import load_dotenv
from datasets import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.inference_engine import InferenceEngine, Hyperparameters
from utils.standard_utilities import load_prompt, extract_content_from_xml_tags
from data_processing.chunk_raw_data import process_files_with_settings
from data_processing.make_dataset import read_chunks, make_huggingface_dataset, push_to_hub
from data_processing.make_multihop_preparings import generate_multihop_pairings
from generation.generate_single_shot_questions_local import generate_questions as generate_single_shot_questions
from generation.generate_single_shot_questions_local import push_to_huggingface as push_single_shot_questions_to_huggingface
from generation.generate_multihop_questions_local import generate_questions as generate_multihop_questions
from generation.generate_multihop_questions_local import push_to_huggingface as push_multihop_questions_to_huggingface
from datasets import load_dataset
import time

# Load environment variables at startup
load_dotenv()

def setup_logger():
    """Configure loguru logger with appropriate formats and levels."""
    # Remove default handler
    logger.remove()
    
    # Add stdout handler for INFO and above
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add stderr handler for ERROR and above
    logger.add(
        sys.stderr,
        format="<red>{time:YYYY-MM-DD HH:mm:ss}</red> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="ERROR"
    )

def validate_path(path: str, path_type: str, create: bool = False) -> Path:
    """
    Validate that the provided path exists and is a directory.
    Optionally create the directory if it doesn't exist.
    
    Args:
        path: String path to validate
        path_type: Type of path (for error messages)
        create: Whether to create the directory if it doesn't exist
        
    Returns:
        Path object if valid
        
    Raises:
        ValueError: If path is invalid and cannot be created
    """
    try:
        logger.debug(f"Validating {path_type} path: {path}")
        data_path = Path(path)
        
        if not data_path.exists():
            if create:
                logger.info(f"Creating {path_type} directory: {path}")
                data_path.mkdir(parents=True, exist_ok=True)
            else:
                logger.error(f"{path_type} directory does not exist: {path}")
                raise ValueError(f"{path_type} directory does not exist: {path}")
            
        if not data_path.is_dir():
            logger.error(f"{path_type} path is not a directory: {path}")
            raise ValueError(f"{path_type} path is not a directory: {path}")
            
        return data_path
        
    except Exception as e:
        logger.error(f"Error during {path_type} path validation: {str(e)}")
        raise ValueError(f"Error validating {path_type} path: {str(e)}")

def find_valid_files(path: Path) -> List[Path]:
    """Find all valid text/markdown files in directory tree."""
    valid_extensions = {'.md', '.txt', '.markdown'}
    logger.debug(f"Searching for files with extensions: {valid_extensions}")
    
    valid_files = [
        f for f in path.rglob('*') 
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]
    
    if not valid_files:
        raise ValueError(
            f"No markdown or text files found in directory or its subdirectories: {path}\n"
            f"Directory tree must contain files with extensions: {', '.join(valid_extensions)}"
        )
    
    logger.debug(f"Found {len(valid_files)} valid files")
    logger.debug("Sample of found files:")
    for file in valid_files[:5]:
        logger.debug(f"  - {file.relative_to(path)}")
    
    return valid_files

def get_path_from_args_or_env(args: argparse.Namespace, arg_name: str, env_var: str, path_type: str, create: bool = False) -> Path:
    """Get and validate path from either CLI arguments or environment variables."""
    # Try command line argument first
    path_value = getattr(args, arg_name)
    
    if path_value:
        logger.debug(f"Using {path_type} path from CLI argument: {path_value}")
    else:
        # Fall back to environment variable
        path_value = os.getenv(env_var)
        if path_value:
            logger.debug(f"Using {path_type} path from environment variable: {path_value}")
    
    if not path_value:
        logger.error(f"No {path_type} path provided")
        logger.error(
            f"Please either:\n"
            f"  1. Use --{arg_name.replace('_', '-')} argument\n"
            f"  2. Set {env_var} environment variable"
        )
        sys.exit(1)
        
    try:
        return validate_path(path_value, path_type, create)
    except ValueError as e:
        logger.error(f"Path validation failed: {str(e)}")
        sys.exit(1)

def setup_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser with all needed arguments."""
    logger.debug("Setting up argument parser")
    parser = argparse.ArgumentParser(
        description="Generate questions and answers from markdown/text files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
    YOURBENCH_RAW_FILE_PATH: Default path to raw data files if not specified via --raw-data-files
    YOURBENCH_SUMMARY_FILE_PATH: Default path to summary files if not specified via --summary-files
    YOURBENCH_CHUNKS_FILE_PATH: Default path to chunk files if not specified via --chunks-files
    YOURBENCH_BASE_URL: Default base URL for inference API
    YOURBENCH_API_KEY: Default API key for inference API
    YOURBENCH_STRATEGY: Default strategy for inference
    YOURBENCH_MODEL: Default model for inference
    YOURBENCH_CHUNK_MODEL: Default model for semantic chunking
    YOURBENCH_CHUNK_SIMILARITY_THRESHOLD: Default similarity threshold for chunking
    YOURBENCH_CHUNK_MIN_TOKENS: Default minimum tokens per chunk
    YOURBENCH_CHUNK_MAX_TOKENS: Default maximum tokens per chunk
    YOURBENCH_CHUNK_TARGET_SIZE: Default target chunk size
    YOURBENCH_DATASET_NAME: Default name for the HuggingFace dataset
    YOURBENCH_ORGANIZATION: Default HuggingFace organization name
        """
    )
    
    # Path arguments
    parser.add_argument(
        '--raw-data-files',
        help=(
            'Directory containing raw markdown/text files for question generation. '
            'If not provided, will check YOURBENCH_RAW_FILE_PATH environment variable.'
        ),
        type=str
    )
    
    parser.add_argument(
        '--summary-files',
        help=(
            'Directory for summary files. '
            'If not provided, will check YOURBENCH_SUMMARY_FILE_PATH environment variable.'
        ),
        type=str
    )
    
    # Inference settings
    parser.add_argument(
        '--base-url',
        help='Base URL for inference API. If not provided, will check YOURBENCH_BASE_URL',
        type=str
    )
    
    parser.add_argument(
        '--api-key',
        help='API key for inference API. If not provided, will check YOURBENCH_API_KEY',
        type=str
    )
    
    parser.add_argument(
        '--strategy',
        help='Strategy for inference. If not provided, will check YOURBENCH_STRATEGY',
        type=str
    )

    parser.add_argument(
        '--model',
        help='Model to use for inference. If not provided, will check YOURBENCH_MODEL',
        type=str
    )
    
    # Chunking arguments
    parser.add_argument(
        '--chunks-files',
        help=(
            'Directory for chunk files. '
            'If not provided, will check YOURBENCH_CHUNKS_FILE_PATH environment variable.'
        ),
        type=str
    )
    
    parser.add_argument(
        '--chunk-model',
        help='Model to use for semantic chunking. If not provided, will check YOURBENCH_CHUNK_MODEL',
        type=str,
        default='all-mpnet-base-v2'
    )
    
    parser.add_argument(
        '--chunk-similarity-threshold',
        help='Similarity threshold for chunking. If not provided, will check YOURBENCH_CHUNK_SIMILARITY_THRESHOLD',
        type=float,
        default=0.9
    )
    
    parser.add_argument(
        '--chunk-min-tokens',
        help='Minimum tokens per chunk. If not provided, will check YOURBENCH_CHUNK_MIN_TOKENS',
        type=int,
        default=256
    )
    
    parser.add_argument(
        '--chunk-max-tokens',
        help='Maximum tokens per chunk. If not provided, will check YOURBENCH_CHUNK_MAX_TOKENS',
        type=int,
        default=1024
    )
    
    parser.add_argument(
        '--chunk-target-size',
        help='Target chunk size. If not provided, will check YOURBENCH_CHUNK_TARGET_SIZE',
        type=int,
        default=512
    )

    parser.add_argument(
        '--debug',
        help='Enable debug mode',
        action='store_true'
    )

    parser.add_argument(
        '--dataset-name',
        help='Name for the HuggingFace dataset. If not provided, will check YOURBENCH_DATASET_NAME',
        type=str
    )
    
    parser.add_argument(
        '--organization',
        help='HuggingFace organization name. If not provided, will check YOURBENCH_ORGANIZATION',
        type=str
    )

    parser.add_argument(
        '--private',
        help='Whether to make the dataset private',
        action='store_true',
        default=True
    )

    parser.add_argument('--question-types', nargs='+', 
                    default=["analytical", "application-based", "clarification", 
                            "conceptual", "counterfactual", "edge-case", "factual", 
                            "false-premise", "open-ended", "true-false"],
                    help='List of question types to generate')
    
    parser.add_argument(
        '--split',
        help='Dataset split to use',
        type=str,
        default='train'
    )

    parser.add_argument(
        '--output-dataset',
        help='Output dataset name on HuggingFace',
        type=str
    )

    parser.add_argument(
        '--dataset',
        help='Dataset name',
        required=False,
        default= os.getenv("YOURBENCH_ORGANIZATION") + "/" + os.getenv("YOURBENCH_DATASET_NAME")
    )

    parser.add_argument(
        '--max-concurrent',
        help='Maximum number of concurrent question generation tasks',
        type=int,
        default=5
    )

    return parser

def process_dataset(
    chunks_path: Path,
    summary_path: Path,
    dataset_settings: Dict[str, str]
) -> Dataset:
    """Create and upload dataset from chunks and summaries."""
    logger.info("Starting dataset creation process...")
    
    try:
        # Read chunks and create dataset
        documents = read_chunks(str(chunks_path), str(summary_path))
        dataset = make_huggingface_dataset(documents)
        
        logger.info(f"Created dataset with {len(dataset)} chunks")
        
        # Push to hub
        push_to_hub(
            dataset,
            dataset_settings['dataset_name'],
            dataset_settings['organization'],
            dataset_settings['private']
        )
        
        logger.info(f"Dataset pushed to HuggingFace Hub: {dataset_settings['organization']}/{dataset_settings['dataset_name']}")
        return dataset
        
    except Exception as e:
        logger.error(f"Error during dataset creation: {str(e)}")
        raise

def create_multihop_dataset(
    base_dataset: Dataset,
    dataset_settings: Dict[str, str]
) -> None:
    """Create and upload multi-hop version of the dataset."""
    logger.info("Starting multi-hop dataset creation...")
    
    try:
        # Create dataset dict structure expected by generate_multihop_pairings
        dataset_dict = {'train': base_dataset}
        
        # Generate multi-hop pairings
        multihop_dataset = generate_multihop_pairings(dataset_dict)
        
        # Push to hub with -multihop suffix
        multihop_name = f"{dataset_settings['dataset_name']}-multihop"
        push_to_hub(
            multihop_dataset,
            multihop_name,
            dataset_settings['organization'],
            dataset_settings['private']
        )
        
        logger.info(f"Multi-hop dataset pushed to HuggingFace Hub: {dataset_settings['organization']}/{multihop_name}")
        
    except Exception as e:
        logger.error(f"Error during multi-hop dataset creation: {str(e)}")
        raise

def get_dataset_settings(args: argparse.Namespace) -> Dict[str, str]:
    """Get dataset settings from args or environment variables."""
    settings = {}
    
    # Map of argument names to environment variables and default values
    setting_map = {
        'dataset_name': 'YOURBENCH_DATASET_NAME',
        'organization': 'YOURBENCH_ORGANIZATION',
    }
    
    for arg_name, env_var in setting_map.items():
        value = getattr(args, arg_name) or os.getenv(env_var)
        if not value:
            logger.error(f"Missing required dataset setting: {arg_name}")
            logger.error(
                f"Please either:\n"
                f"  1. Use --{arg_name.replace('_', '-')} argument\n"
                f"  2. Set {env_var} environment variable"
            )
            sys.exit(1)
        settings[arg_name] = value
        logger.debug(f"Using {arg_name}: {value}")
    
    settings['private'] = args.private
    return settings

def get_single_shot_question_settings(args: argparse.Namespace) -> Dict[str, str]:
    """Get single-shot question settings from args or environment variables."""
    settings = {}
    
    # Map of argument names to environment variables and default values
    setting_map = {
        'dataset_name': 'YOURBENCH_DATASET_NAME',
        'organization': 'YOURBENCH_ORGANIZATION',
    }
    
    for arg_name, env_var in setting_map.items():
        value = getattr(args, arg_name) or os.getenv(env_var)
        if not value:
            logger.error(f"Missing required dataset setting: {arg_name}")
            logger.error(
                f"Please either:\n"
                f"  1. Use --{arg_name.replace('_', '-')} argument\n"
                f"  2. Set {env_var} environment variable"
            )
            sys.exit(1)
        settings[arg_name] = value
        logger.debug(f"Using {arg_name}: {value}")
    
    settings['private'] = args.private
    settings['output_dataset'] = f"{settings['organization']}/{settings['dataset_name']}-single-shot-questions"
    return settings

def get_inference_settings(args: argparse.Namespace) -> Dict[str, str]:
    """Get inference settings from args or environment variables."""
    settings = {}
    
    # Map of argument names to environment variables
    setting_map = {
        'base_url': 'YOURBENCH_BASE_URL',
        'api_key': 'YOURBENCH_API_KEY',
        'strategy': 'YOURBENCH_STRATEGY',
        'model': 'YOURBENCH_MODEL'
    }
    
    for arg_name, env_var in setting_map.items():
        value = getattr(args, arg_name) or os.getenv(env_var)
        if not value:
            logger.error(f"Missing required inference setting: {arg_name}")
            logger.error(
                f"Please either:\n"
                f"  1. Use --{arg_name.replace('_', '-')} argument\n"
                f"  2. Set {env_var} environment variable"
            )
            sys.exit(1)
        settings[arg_name] = value
        logger.debug(f"Using {arg_name}: {value}")
    
    return settings

def mirror_directory_structure(
    raw_files: List[Path],
    raw_path: Path,
    summary_path: Path,
) -> Dict[Path, Path]:
    """
    Mirror the directory structure from raw_path to summary_path.
    Returns mapping of raw files to their corresponding summary files.
    """
    file_mapping = {}
    
    for raw_file in raw_files:
        # Get the relative path to maintain directory structure
        relative_path = raw_file.relative_to(raw_path)
        summary_file = summary_path / relative_path
        
        # Create parent directories if they don't exist
        if not summary_file.parent.exists():
            logger.info(f"Creating directory structure: {summary_file.parent}")
            summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create empty summary file if it doesn't exist
        if not summary_file.exists():
            logger.info(f"Creating empty summary file: {summary_file}")
            summary_file.touch()
        
        file_mapping[raw_file] = summary_file
        
    return file_mapping

def generate_summary(raw_file: Path, summary_file: Path, inference_settings: Dict[str, str]) -> None:
    """Generate a summary for a raw file."""
    # make an inference engine
    inference_engine = InferenceEngine(
        model_name=inference_settings['model'],
        connection_details = {
            'strategy': inference_settings['strategy'],
            'base_url': inference_settings['base_url'],
            'api_key': inference_settings['api_key'],
        }
    )
    # read the raw file
    with open(raw_file, 'r') as f:
        document = f.read()

    # read the prompt
    with open('prompts/summarize.md', 'r') as f:
        prompt = f.read()
    
    prompt = prompt.format(document=document)

    # make an inference
    raw_response = inference_engine.single_inference(
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )[0]
    
    # extract the summary
    summary = extract_content_from_xml_tags(raw_response, 'final_summary')

    # check if the summary is empty
    if not summary:
        # set the summary to:
        summary = "This document does not have a summary."
    
    # write the summary to the summary file
    with open(summary_file, 'w') as f:
        f.write(summary)

def check_and_generate_summaries(
    raw_files: List[Path],
    raw_path: Path,
    summary_path: Path,
    inference_settings: Dict[str, str]
) -> None:
    """Check for missing summaries and generate them as needed."""
    logger.info("Setting up summary directory structure...")
    
    # First, mirror the directory structure and get file mapping
    file_mapping = mirror_directory_structure(raw_files, raw_path, summary_path)
    
    logger.info("Checking for missing or empty summaries...")
    for raw_file, summary_file in file_mapping.items():
        # Check if summary is empty or needs updating
        if summary_file.stat().st_size == 0:
            relative_path = raw_file.relative_to(raw_path)
            logger.info(f"Missing or empty summary for: {relative_path}")
            logger.debug(f"Will generate summary at: {summary_file}")
            
            try:
                # TODO: Implement or import generate_summary function
                generate_summary(raw_file, summary_file, inference_settings)
                logger.info(f"Generated summary for: {relative_path}")
            except Exception as e:
                logger.error(f"Failed to generate summary for {relative_path}: {str(e)}")
        else:
            logger.debug(f"Summary exists for: {raw_file.relative_to(raw_path)}")

def get_chunking_settings(args: argparse.Namespace) -> Dict[str, any]:
    """Get chunking settings from args or environment variables."""
    settings = {}
    
    # Map of argument names to environment variables and default values
    setting_map = {
        'chunk_model': ('YOURBENCH_CHUNK_MODEL', 'all-mpnet-base-v2'),
        'chunk_similarity_threshold': ('YOURBENCH_CHUNK_SIMILARITY_THRESHOLD', 0.9),
        'chunk_min_tokens': ('YOURBENCH_CHUNK_MIN_TOKENS', 256),
        'chunk_max_tokens': ('YOURBENCH_CHUNK_MAX_TOKENS', 1024),
        'chunk_target_size': ('YOURBENCH_CHUNK_TARGET_SIZE', 512)
    }
    
    for arg_name, (env_var, default_value) in setting_map.items():
        # Try command line argument first
        value = getattr(args, arg_name)
        
        # Then environment variable
        if value is None:
            value = os.getenv(env_var)
            
        # Finally default value
        if value is None:
            value = default_value
            
        # Convert to correct type based on default value type
        if isinstance(default_value, float):
            value = float(value)
        elif isinstance(default_value, int):
            value = int(value)
            
        settings[arg_name] = value
        logger.debug(f"Using {arg_name}: {value}")
    
    return settings

def process_chunks(
    raw_files: List[Path],
    raw_path: Path,
    chunks_path: Path,
    chunking_settings: Dict[str, any]
) -> None:
    """Process raw files into semantic chunks."""
    logger.info("Starting semantic chunking process...")
    
    # Create settings list for chunk_raw_data
    settings_list = [{
        'similarity_threshold': chunking_settings['chunk_similarity_threshold'],
        'min_tokens': chunking_settings['chunk_min_tokens'],
        'max_tokens': chunking_settings['chunk_max_tokens'],
        'target_chunk_size': chunking_settings['chunk_target_size']
    }]
    
    try:
        # Process files using existing function
        settings_chunk_lengths = process_files_with_settings(
            str(raw_path),
            str(chunks_path),
            settings_list,
            chunking_settings['chunk_model']
        )
        
        # Log chunking results
        for result in settings_chunk_lengths:
            settings = result['settings']
            mean_length = result['mean']
            stdev_length = result['stdev']
            variance_length = result['variance']
            chunk_count = len(result['chunk_lengths'])
            
            logger.info("Chunking Results:")
            logger.info(f"Settings: {settings}")
            logger.info(f"Mean chunk length: {mean_length:.2f} tokens")
            logger.info(f"Standard deviation: {stdev_length:.2f}")
            logger.info(f"Variance: {variance_length:.2f}")
            logger.info(f"Number of chunks: {chunk_count}")
            
        logger.info("Semantic chunking completed successfully")
        
    except Exception as e:
        logger.error(f"Error during semantic chunking: {str(e)}")
        raise

def main():
    """Main entry point for the question generation library."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging based on debug flag
    setup_logger()
    if args.debug:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="DEBUG"
        )
    
    logger.info("Starting question generation process")
    
    # Get and validate paths
    raw_data_path = get_path_from_args_or_env(
        args, 'raw_data_files', 'YOURBENCH_RAW_FILE_PATH', 'raw data'
    )
    # For summary path, we'll create it if it doesn't exist
    summary_path = get_path_from_args_or_env(
        args, 'summary_files', 'YOURBENCH_SUMMARY_FILE_PATH', 'summary', create=True
    )

    chunks_path = get_path_from_args_or_env(
        args, 'chunks_files', 'YOURBENCH_CHUNKS_FILE_PATH', 'chunks', create=True
    )
    
    # Get inference settings
    inference_settings = get_inference_settings(args)

    # make an inference engine
    engine = InferenceEngine(
        model_name=inference_settings['model'],
        connection_details = {
            'strategy': inference_settings['strategy'],
            'base_url': inference_settings['base_url'],
            'api_key': inference_settings['api_key'],
        }
    )
    
    # Find all valid files
    raw_files = find_valid_files(raw_data_path)
    
    # Check and generate summaries
    check_and_generate_summaries(raw_files, raw_data_path, summary_path, inference_settings)

    # Get chunking settings
    chunking_settings = get_chunking_settings(args)
    
    # Process semantic chunks
    process_chunks(raw_files, raw_data_path, chunks_path, chunking_settings)

    dataset_settings = get_dataset_settings(args)

    base_dataset = process_dataset(chunks_path, summary_path, dataset_settings)

    create_multihop_dataset(base_dataset, dataset_settings)

    single_shot_question_settings = get_single_shot_question_settings(args)

    # load the datasets
    # document_dataset = load_dataset(args.dataset, split=args.split)
    # for question_type in args.question_types:
    #     start_time = time.time()
    #     document_dataset_with_questions = generate_single_shot_questions(document_dataset, engine, question_type, args.max_concurrent)
    #     end_time = time.time()
    #     logger.info(f"Generated {len(document_dataset_with_questions)} questions for {question_type} in {end_time - start_time:.2f} seconds")
    #     push_single_shot_questions_to_huggingface(document_dataset_with_questions, single_shot_question_settings['output_dataset'])

    logger.info("Single-shot questions generated successfully")
    
    # now we touch multihop reasoning
    multihop_dataset = load_dataset(f"{dataset_settings['organization']}/{dataset_settings['dataset_name']}-multihop", split=args.split)
    for question_type in args.question_types:
        start_time = time.time()
        multihop_dataset_with_questions = generate_multihop_questions(multihop_dataset, engine, question_type, args.max_concurrent)
        end_time = time.time()
        logger.info(f"Generated {len(multihop_dataset_with_questions)} questions for {question_type} in {end_time - start_time:.2f} seconds")
        push_multihop_questions_to_huggingface(multihop_dataset_with_questions, single_shot_question_settings['output_dataset'])

    
    logger.info("Process completed successfully")

if __name__ == "__main__":
    main()