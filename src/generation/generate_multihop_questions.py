import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import json
import re

import numpy as np

from typing import List, Dict, Any
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from dotenv import load_dotenv
from collections import defaultdict

from utils.file_utilities import load_prompt
from utils.inference_engine import InferenceEngine, Hyperparameters
from models.integrative_qa_pair import GeneratedIntegrativeQAPair, ChunkAnalysis
from loguru import logger

# load environment variables
load_dotenv()

BETTERBENCH_MULTIHOP_PAIRINGS_DATASET = os.getenv("BETTERBENCH_MULTIHOP_PAIRINGS_DATASET")
BETTERBENCH_MULTIHOP_QUESTIONS_DATASET = os.getenv("BETTERBENCH_MULTIHOP_QUESTIONS_DATASET")

user_prompt_template = load_prompt("question_generation/multi_hop/user_prompt_template")

def extract_content_from_xml_tags(full_content, xml_tag):
    # This function extracts the content between the XML tags
    # It uses regex to find the content and includes error handling

    # Define the regex patterns to match the content
    pattern_with_closing_tag = f"<{xml_tag}>(.*?)</{xml_tag}>"
    pattern_without_closing_tag = f"<{xml_tag}>(.*)"

    try:
        # First, try to find matches with both opening and closing tags
        matches_with_closing = re.findall(
            pattern_with_closing_tag, full_content, re.DOTALL
        )
        if matches_with_closing:
            return matches_with_closing[0].strip()

        # If no matches found, try to find content with only opening tag
        matches_without_closing = re.findall(
            pattern_without_closing_tag, full_content, re.DOTALL
        )
        if matches_without_closing:
            return matches_without_closing[0].strip()

        # If still no matches found, return an empty string
        return ""

    except Exception as extraction_error:
        print(f"Error extracting content from XML tags: {extraction_error}")
        return ""

def create_empty_qa_pair() -> Dict:
    return {
        "document_analysis": "",
        "chunks_analysis": [],
        "integration_points": [],
        "potential_integrative_questions": [],
        "best_direction": "",
        "direct_line_quotes": {},
        "question": "",
        "answer": "",
        "reasoning": "",
        "chunks_used": ["none"],
        "kind": "invalid_response",
        "estimated_difficulty": 1
    }

def parse_and_validate_qa_response(result: str) -> Dict:
    """
    Parse and validate a QA response, handling both single objects and lists.
    
    Args:
        result (str): JSON string from the model
        
    Returns:
        Dict: Validated QA pair or empty template if validation fails
    """
    try:
        # Parse the JSON response
        parsed_data = json.loads(result)
        
        # Handle both list and single object responses
        if isinstance(parsed_data, list):
            # Take the first item if it's a list
            if len(parsed_data) > 0:
                qa_data = parsed_data[0]
            else:
                logger.warning("Received empty list in response")
                return create_empty_qa_pair()
        else:
            qa_data = parsed_data
            
        # Validate with Pydantic
        validated_qa = GeneratedIntegrativeQAPair(**qa_data)
        return validated_qa.dict()
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse/validate QA pair: {str(e)}")
        return create_empty_qa_pair()
        
    except Exception as e:
        logger.error(f"Unexpected error processing QA pair: {str(e)}")
        return create_empty_qa_pair()

def generate_multihop_questions_for_prompt(dataset: Dataset, prompt: str, inference_engine: InferenceEngine) -> Dataset:
    messages = []
    for example in dataset:
        # Format chunks into a readable format
        chunks_formatted = "\n\n".join([
            f"CHUNK {i+1}:\n{chunk}" 
            for i, chunk in enumerate(example['chunks'])
        ])
        
        # Create the full prompt with context
        context = {
            "summary": example['summary'],
            "title": example['title'],
            "chunks": chunks_formatted
        }
        
        # Format the prompt with the context
        formatted_prompt = user_prompt_template.format(
            summary=context['summary'],
            title=context['title'],
            chunks=context['chunks']
        )
        
        # Create message for inference
        message = [{"role" : "system", "content" : prompt},{"role": "user", "content": formatted_prompt}]
        messages.append(message)
    
    
    results, usages = inference_engine.parallel_inference(
        messages=messages,
        hyperparameters=Hyperparameters(temperature=1),
        max_concurrent_requests=1024
    )

    print(results[0])

    # Parse and validate results
    generated_qa_pairs = [
        parse_and_validate_qa_response(extract_content_from_xml_tags(result, "generated_questions"))
        for result in results
    ]
    
    # Create new dataset with original features and generated questions
    new_features = {}
    
    # Copy original features
    for key in dataset.features:
        new_features[key] = [example[key] for example in dataset]
    
    # Add generated features
    qa_fields = [
        "document_analysis", "chunks_analysis", "integration_points",
        "potential_integrative_questions", "best_direction", 
        "direct_line_quotes", "question", "answer", "reasoning",
        "chunks_used", "kind", "estimated_difficulty"
    ]
    
    for field in qa_fields:
        new_features[f"generated_{field}"] = [
            qa_pair.get(field, create_empty_qa_pair()[field]) 
            for qa_pair in generated_qa_pairs
        ]
    
    # Create and return new dataset
    return Dataset.from_dict(new_features)


if __name__ == "__main__":
    dataset = load_dataset(BETTERBENCH_MULTIHOP_PAIRINGS_DATASET, split="train")
    inference_engine = InferenceEngine(
        connection_details={
            "api_key" : "EMPTY",
            "base_url" : "http://localhost:30000/v1/",
            "strategy" : "openai"
        },
        model_name="mistralai/Mistral-Large-Instruct-2407"
    )
    basic_factual_prompt = load_prompt("question_generation/multi_hop/generate_simple_factual_multihop")

    prompts = [basic_factual_prompt]

    enhanced_dataset = generate_multihop_questions_for_prompt(dataset, prompts[0], inference_engine)
    enhanced_dataset.push_to_hub(BETTERBENCH_MULTIHOP_QUESTIONS_DATASET, private=True)
    for prompt in prompts[1:]:
        generated_questions = generate_multihop_questions_for_prompt(dataset, prompt, inference_engine)
        enhanced_dataset = concatenate_datasets([enhanced_dataset, generated_questions])
        enhanced_dataset.push_to_hub(BETTERBENCH_MULTIHOP_QUESTIONS_DATASET, private=True)
