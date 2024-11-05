# ===
# This is a version without the window based contextualization
# ===

import sys
import os
import argparse
import socket
import psutil
import subprocess
import signal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset, load_dataset, concatenate_datasets
from pydantic import BaseModel, Field, ValidationError

from utils.inference_engine import InferenceEngine, Hyperparameters
from loguru import logger
from typing import Dict, List
from huggingface_hub import HfApi
from enum import Enum
from typing import List
from pydantic import BaseModel, Field, constr
from utils.file_utilities import load_prompt
from utils.standard_utilities import extract_content_from_xml_tags
import copy
import json
from pathlib import Path
import time

row_structure = {
        "chunk_uuid" : "",
        # everything is done by the generator model
        "generator_model" : "",
        "question_type" : "",
        "question" : "",
        "answer" : "",
        "document_analysis" : "",
        "chunk_analysis" : "",
        "potential_question_directions" : "",
        "best_direction" : "",
        "direct_quotes" : "",
        "reasoning" : "",
        "estimated_difficulty" : "",
        "testable_concepts" : "",
        "quality_metrics" : "",
        "difficulty_justification" : "",
        "quote_context" : "",
        "supporting_quotes" : "",
        "chunk_analysis" : "",
    }


class QuestionType(str, Enum):
    ANALYTICAL = "analytical"  # Questions requiring analysis and synthesis
    APPLICATION = "application"  # Questions requiring application of text concepts
    CLARIFICATION = "clarification"  # Questions seeking deeper understanding of specific elements
    CONCEPTUAL = "conceptual"  # Questions testing understanding of principles and ideas
    COUNTERFACTUAL = "counterfactual"  # Questions exploring alternative scenarios
    EDGE_CASE = "edge-case"  # Questions exploring edge cases and exceptions
    FACTUAL = "factual"  # Questions requiring specific fact recall from text
    FALSE_PREMISE = "false-premise"  # Questions presenting incorrect assumptions for correction
    FALSE_PREMISE_2 = "False-premise"  # Questions presenting incorrect assumptions for correction
    OPEN_ENDED = "open-ended"  # Questions encouraging exploration and discussion
    TRUE_FALSE = "true-false"
    TRUE_FALSE_2 = "True-False"

class DifficultyLevel(int, Enum):
    VERY_EASY = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    VERY_HARD = 5


class QuestionQuality(BaseModel):
    clear_language: bool = Field(...)
    text_based: bool = Field(...)
    no_tricks: bool = Field(...)

class GeneratedQuestionAnswerPair(BaseModel):
    document_extract_analysis: str = Field(..., min_length=10)
    testable_concepts: List[str] = Field(..., min_items=2)
    potential_question_directions: List[str] = Field(..., min_items=2)
    best_direction: str = Field(..., min_length=10)
    comprehension_type: str = Field(...)
    quality_metrics: QuestionQuality = Field(...)
    supporting_quotes: List[str] = Field(..., min_items=1)
    quote_context: str = Field(..., min_length=10)
    kind: QuestionType = Field(...)
    question: str = Field(...)
    answer: str = Field(...)
    reasoning: str = Field(..., min_length=10)
    difficulty: DifficultyLevel = Field(...)
    difficulty_justification: str = Field(..., min_length=30)
    
    class Config:
        enum_values = True
        
        
def make_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
def parse_results(content: str) -> tuple[List[GeneratedQuestionAnswerPair], int]:
    """Parse and validate question results from content string."""
    logger.debug(f"Starting to parse content of length: {len(content)}")
    total_candidates = 0
    validated_list = []
    
    try:
        # Replace JavaScript true/false with Python True/False
        content = content.replace('true', 'True').replace('false', 'False')
        extracted_content = eval(content)
        total_candidates = len(extracted_content) if extracted_content else 0
        logger.debug(f"Successfully extracted {total_candidates} candidate questions")
        
        try:
            for idx, question in enumerate(extracted_content):
                try:
                    validated_question = GeneratedQuestionAnswerPair(**question)
                    validated_list.append(validated_question)
                    logger.debug(f"Validated question {idx+1}/{total_candidates}")
                except ValidationError as e:
                    logger.error(f"Validation error for question {idx+1}: {str(e)}")
            logger.info(f"Successfully validated {len(validated_list)}/{total_candidates} questions")
            return validated_list, total_candidates
        except Exception as e:
            logger.error(f"Error parsing questions: {str(e)}")
            return validated_list, total_candidates
    except Exception as e:
        logger.error(f"Error evaluating content: {str(e)}")
        return [], 0

def generate_questions(document_dataset: Dataset, engine: InferenceEngine, question_type: str, max_concurrent: int = 1024) -> Dataset:
    """Generate questions for a given dataset."""
    logger.info(f"Starting question generation for type: {question_type}")
    logger.info(f"Processing {len(document_dataset)} document chunks")
    
    # Initialize statistics
    generation_stats = {
        "generator_model": engine.model_name,
        "question_type": question_type,
        "total_processed_chunks": len(document_dataset),
        "total_generated_responses": 0,
        "total_extracted_valid_xml": 0,
        "total_generated_questions": 0,
        "total_valid_jsons": 0
    }
    
    logger.debug("Loading prompts...")
    prompt = load_prompt(f"question_generation/single_shot/{question_type}")
    user_template = load_prompt("question_generation/single_shot/_user_template")
    logger.info(f"Loaded prompts for {question_type}")
    
    logger.debug("Preparing message list...")
    message_list = []
    for idx, row in enumerate(document_dataset):
        user_message = user_template.format(document_summary=row["summary"], chunk=row["chunk"])
        message_list.append(make_messages(prompt, user_message))
        if idx % 100 == 0:
            logger.debug(f"Prepared {idx}/{len(document_dataset)} messages")
    
    logger.info(f"Starting parallel inference for {len(message_list)} messages")
    inference_results = engine.parallel_inference(message_list, max_concurrent_requests=max_concurrent)[0]
    generation_stats["total_generated_responses"] = len([r for r in inference_results if r and r.strip()])
    logger.info(f"Completed inference with {generation_stats['total_generated_responses']} responses")
    
    logger.debug("Extracting content from XML tags...")
    extracted_content = [extract_content_from_xml_tags(result, "generated_questions") for result in inference_results]
    generation_stats["total_extracted_valid_xml"] = len([c for c in extracted_content if c and c.strip()])
    logger.info(f"Successfully extracted {generation_stats['total_extracted_valid_xml']} valid XML contents")
    
    logger.debug("Parsing results...")
    parsed_results_and_counts = [parse_results(content) for content in extracted_content]
    parsed_results = [result[0] for result in parsed_results_and_counts]
    
    # Update statistics
    generation_stats["total_generated_questions"] = sum(count for _, count in parsed_results_and_counts)
    generation_stats["total_valid_jsons"] = sum(len(result) for result in parsed_results)
    
    # Write statistics to JSONL file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "generation_stats.jsonl"
    
    with open(log_file, "a") as f:
        f.write(json.dumps(generation_stats) + "\n")
    
    logger.info(f"Generation statistics: {generation_stats}")
    
    logger.debug("Unwrapping results into dataset format...")
    unwrapped_results = []
    for idx, (dataset_item, parsed_result) in enumerate(zip(document_dataset, parsed_results)):
        chunk_uuid = dataset_item["chunk_uuid"]
        generator_model = engine.model_name
        for question in parsed_result:
            prepared_row = copy.deepcopy(row_structure)
            prepared_row["chunk_uuid"] = chunk_uuid
            prepared_row["generator_model"] = generator_model
            prepared_row["question_type"] = question.kind
            prepared_row["question"] = question.question
            prepared_row["answer"] = question.answer
            prepared_row["document_analysis"] = question.document_extract_analysis
            prepared_row["potential_question_directions"] = list(question.potential_question_directions)
            prepared_row["best_direction"] = question.best_direction
            prepared_row["direct_quotes"] = list(question.supporting_quotes)
            prepared_row["quote_context"] = question.quote_context
            prepared_row["reasoning"] = question.reasoning
            prepared_row["estimated_difficulty"] = int(question.difficulty)
            prepared_row["difficulty_justification"] = question.difficulty_justification
            prepared_row["testable_concepts"] = list(question.testable_concepts)
            prepared_row["quality_metrics"] = question.quality_metrics.json()
            prepared_row["supporting_quotes"] = list(question.supporting_quotes)
            unwrapped_results.append(prepared_row)
        if idx % 100 == 0:
            logger.debug(f"Unwrapped {idx}/{len(document_dataset)} chunks")
    
    logger.info(f"Created final dataset with {len(unwrapped_results)} questions")
    unwrapped_dataset = Dataset.from_list(unwrapped_results)
    return unwrapped_dataset



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
        logger.error(f"Error loading existing dataset: {str(e)}")
        raise

def push_to_huggingface(dataset: Dataset, repo_id: str) -> None:
    """Push the processed logs to HuggingFace as a dataset."""
    logger.info(f"Starting push to HuggingFace repo: {repo_id}")
    logger.info(f"New dataset contains {len(dataset)} entries")
    
    # Add check for empty dataset
    if len(dataset) == 0:
        logger.warning("Skipping push to HuggingFace - dataset is empty")
        return
        
    try:
        if dataset_exists(repo_id):
            logger.info("Found existing dataset. Loading and concatenating...")
            try:
                existing_dataset = load_existing_dataset(repo_id)
                if existing_dataset is not None and len(existing_dataset) > 0:
                    logger.info(f"Loaded existing dataset with {len(existing_dataset)} entries")
                    combined_dataset = concatenate_datasets([existing_dataset, dataset])
                    logger.info(f"Combined dataset size: {len(combined_dataset)} entries")
                else:
                    logger.warning("Existing dataset is empty or invalid. Using only new data.")
                    combined_dataset = dataset
            except Exception as e:
                logger.warning(f"Error loading existing dataset: {str(e)}. Using only new data.")
                combined_dataset = dataset
        else:
            logger.info("No existing dataset found. Creating new dataset...")
            combined_dataset = dataset
        
        logger.info("Pushing to HuggingFace Hub...")
        # Add explicit feature schema using Features class
        from datasets.features import Features, Value, Sequence

        features = Features({
            "chunk_uuid": Value("string"),
            "generator_model": Value("string"),
            "question_type": Value("string"),
            "question": Value("string"),
            "answer": Value("string"),
            "document_analysis": Value("string"),
            "chunk_analysis": Value("string"),
            "potential_question_directions": Sequence(Value("string")),
            "best_direction": Value("string"),
            "direct_quotes": Sequence(Value("string")),
            "reasoning": Value("string"),
            "estimated_difficulty": Value("int64"),
            "testable_concepts": Sequence(Value("string")),
            "quality_metrics": Value("string"),
            "difficulty_justification": Value("string"),
            "quote_context": Value("string"),
            "supporting_quotes": Sequence(Value("string")),
        })
        
        combined_dataset.to_json("temp_dataset.json")  # Save to temporary file
        combined_dataset = Dataset.from_json(
            "temp_dataset.json",
            features=features
        )
        
        combined_dataset.push_to_hub(repo_id, private=True)
        logger.success(f"Successfully pushed dataset to {repo_id}")
        
        # Cleanup
        import os
        if os.path.exists("temp_dataset.json"):
            os.remove("temp_dataset.json")
            
    except Exception as e:
        logger.error(f"Error during push to HuggingFace: {str(e)}")
        raise

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_openai_server(model_name: str) -> subprocess.Popen:
    if is_port_in_use(3000):
        raise RuntimeError("Port 3000 is already in use. Cannot start the OpenAI server.")
    
    cmd = f"vllm serve {model_name} --trust-remote-code --dtype auto --port 3000 --tensor-parallel-size 2 --gpu-memory-utilization 0.94 --enable-prefix-caching"
    process = subprocess.Popen(cmd, shell=True)
    return process

def terminate_process_tree(pid: int, include_parent: bool = True):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.terminate()
    psutil.wait_procs(children, timeout=5)
    if include_parent:
        parent.terminate()
        parent.wait(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate questions from a dataset')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default="sumuks/y1",
                      help='HuggingFace dataset ID (default: sumuks/y1)')
    parser.add_argument('--split', type=str, default="train",
                      help='Dataset split to use (default: train)')
    parser.add_argument('--output-dataset', type=str, default="sumuks/y1-questions-x4",
                      help='Output dataset ID on HuggingFace (default: sumuks/y1-questions)')
    
    # Question generation arguments
    parser.add_argument('--question-types', nargs='+', 
                      default=["analytical", "application-based", "clarification", 
                              "conceptual", "counterfactual", "edge-case", "factual", 
                              "false-premise", "open-ended", "true-false"],
                      help='List of question types to generate')
    
    # Inference engine arguments
    parser.add_argument('--strategy', type=str, default="openai",
                      help='Inference strategy (default: openai)')
    parser.add_argument('--api-key', type=str, default="EMPTY",
                      help='API key for inference (default: EMPTY)')
    parser.add_argument('--base-url', type=str, default="http://localhost:3000/v1/",
                      help='Base URL for inference (default: http://localhost:3000/v1/)')
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct",
                      help='Model name (default: meta-llama/Meta-Llama-3.1-70B-Instruct)')
    parser.add_argument('--max-concurrent', type=int, default=1024,
                      help='Maximum concurrent requests (default: 1024)')
    parser.add_argument('--start-server', action='store_true',
                      help='Start the OpenAI-compatible server')
    
    args = parser.parse_args()
    
    server_process = None
    try:
        if args.start_server:
            server_process = start_openai_server(args.model)
            # Give the server a moment to start
            time.sleep(300)
        
        document_dataset = load_dataset(args.dataset, split=args.split)
        engine = InferenceEngine(
            connection_details = {
                "strategy": args.strategy,
                "api_key": args.api_key,
                "base_url": args.base_url
            },
            model_name=args.model
        )

        for question_type in args.question_types:
            start_time = time.time()
            document_dataset_with_questions = generate_questions(document_dataset, engine, question_type, args.max_concurrent)
            end_time = time.time()
            logger.info(f"Generated {len(document_dataset_with_questions)} questions for {question_type} in {end_time - start_time:.2f} seconds")
            push_to_huggingface(document_dataset_with_questions, args.output_dataset)
    finally:
        if server_process:
            try:
                terminate_process_tree(server_process.pid)
                logger.info("OpenAI server terminated gracefully")
                
                # Wait for port release
                max_wait = 30
                start = time.time()
                while is_port_in_use(3000) and time.time() - start < max_wait:
                    time.sleep(1)
                
                if is_port_in_use(3000):
                    logger.warning("Port 3000 still in use after graceful shutdown. Forcing termination...")
                    os.kill(server_process.pid, signal.SIGKILL)
                
            except Exception as e:
                logger.error(f"Error during server shutdown: {e}")
                try:
                    os.kill(server_process.pid, signal.SIGKILL)
                except:
                    pass