# yourbench/pipeline/multi_hop_question_generation.py

"""
Multi-Hop Question Generation Module

This module handles the multi-hop question generation stage of the Yourbench pipeline.
It extends the single-shot paradigm by combining multiple chunks of text together 
and generating questions that require synthesizing information across these multiple chunks.

Usage:
    from yourbench.pipeline.multi_hop_question_generation import run

    multi_hop_cfg = {
        "source_dataset_name": "yb_demo_chunked_documents",
        "output_dataset_name": "yb_demo_multi_hop_questions",
        "local_dataset_path": "data/example/multi_hop_questions",
        "concat_existing_dataset": False,
        "run": True
    }
    run({"pipeline": {"multi_hop_question_generation": multi_hop_cfg}, ...})
"""


import json
from typing import Dict, Any, List
from dataclasses import dataclass, field

from loguru import logger

from yourbench.utils.dataset_engine import smart_load_dataset
from yourbench.utils.saving_engine import save_dataset
from yourbench.utils.inference_engine import InferenceCall, run_inference
from yourbench.utils.prompts import (
    MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT,
    MULTI_HOP_QUESTION_GENERATION_USER_PROMPT,
)


@dataclass
class MultiHopQuestion:
    """
    Data structure to hold one generated question-answer pair for multi-hop generation.
    """
    chunk_id: str
    document_id: str
    question: str
    answer: str
    self_estimated_difficulty_rating: int
    self_assessed_question_type: str
    self_answer: str


def run(config: Dict[str, Any]) -> None:
    """
    Main function to run the multi-hop question generation pipeline.

    1. Load the chunked dataset using the configured source_dataset_name.
    2. For each document row, retrieve its multi_hop chunks.
    3. Construct multi-hop question generation prompts that combine all relevant chunks.
    4. Execute inference in batches using the run_inference utility.
    5. Parse the responses, storing them in a new "multi_hop_questions" column.
    6. Save the resulting dataset to disk and optionally push to a Hugging Face dataset.

    Config Requirements:
        pipeline:
          multi_hop_question_generation:
            source_dataset_name: str
            output_dataset_name: str
            local_dataset_path: str
            concat_existing_dataset: bool
            run: bool  # if True, this stage is executed

        model_roles:
          multi_hop_question_generation: [list_of_model_names]
        
        model_list: # array of model configs

    Example:
        pipeline:
          multi_hop_question_generation:
            source_dataset_name: yb_demo_chunked_documents
            output_dataset_name: yb_demo_multi_hop_questions
            local_dataset_path: data/example/multi_hop_questions
            concat_existing_dataset: false
            run: true
    """
    stage_cfg = config.get("pipeline", {}).get("multi_hop_question_generation", {})
    if not stage_cfg.get("run", False):
        logger.info("multi_hop_question_generation stage is disabled. Skipping.")
        return

    # === Load chunked dataset ===
    source_dataset_name = stage_cfg["source_dataset_name"]
    output_dataset_name = stage_cfg["output_dataset_name"]
    logger.info("Loading chunked dataset from: {}", source_dataset_name)

    dataset = smart_load_dataset(source_dataset_name, config)
    logger.info("Loaded dataset with {} rows.", len(dataset))

    # The name(s) of the model(s) used for multi-hop generation:
    generating_model_name = config["model_roles"]["multi_hop_question_generation"][0]

    # We will collect all calls to inference into a list, so we can run them in parallel
    all_inference_calls: List[InferenceCall] = []
    # We also map call-index to row index for reconstruction
    call_index_to_row_id = []

    # Prepare the system prompt once
    system_message = {
        "role": "system",
        "content": MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT
    }

    # The user prompt template will be something like:
    # MULTI_HOP_QUESTION_GENERATION_USER_PROMPT = """
    # <title>{title}</title>
    # <document_summary>{document_summary}</document_summary>
    # <text_chunks>{chunks}</text_chunks>
    # <test_audience>{test_audience}</test_audience>
    # """
    # We'll fill in {chunks} with repeated <text_chunk_i>...</text_chunk_i> blocks.

    for row_idx, row_data in enumerate(dataset):
        # Because we're generating multi-hop questions, the relevant chunks are in "multihop_chunks"
        # which is a list of strings (each item is a concatenation of multiple single-hop chunks).
        # But we can just feed them as one big array in the prompt if we want multiple multi-hop sets.
        # For simplicity, let's do one user prompt per row, bundling all multi-hop chunks in one prompt.
        multi_hop_chunks = row_data.get("multihop_chunks", [])
        if not multi_hop_chunks:
            # If no multi-hop chunks, skip or store empty
            continue

        # Construct the <text_chunks> block
        chunk_blocks = []
        for i, chunk_text in enumerate(multi_hop_chunks):
            block = f"<text_chunk_{i}>{chunk_text}</text_chunk_{i}>"
            chunk_blocks.append(block)
        chunk_section = "\n".join(chunk_blocks)

        title = row_data.get("document_filename", f"Document_{row_idx}")
        doc_summary = row_data.get("document_summary", "No summary provided.")
        test_audience = stage_cfg.get("test_audience", "undergraduate")

        user_prompt_filled = MULTI_HOP_QUESTION_GENERATION_USER_PROMPT.format(
            title=title,
            document_summary=doc_summary,
            chunks=chunk_section,
            test_audience=test_audience
        )
        user_message = {"role": "user", "content": user_prompt_filled}

        # We only do one call per row if we treat all multi-hop chunks as a single block,
        # so let's build that single InferenceCall:
        inference_call = InferenceCall(
            messages=[system_message, user_message],
            tags=["multi_hop_qa"]
        )
        all_inference_calls.append(inference_call)
        call_index_to_row_id.append(row_idx)

    logger.info("Generated {} calls to the inference engine for multi-hop generation.", len(all_inference_calls))

    # === Run inference in a single batch ===
    model_responses_dict = run_inference(
        config=config,
        step_name="multi_hop_question_generation",
        inference_calls=all_inference_calls
    )

    # Because we only used the first model in model_roles, let's just extract that
    model_responses = model_responses_dict[generating_model_name]

    # We'll store the final multi-hop QAs in a list of lists, parallel to the dataset
    # i.e. multi_hop_qa_per_row[row_idx] -> list of QAs
    multi_hop_qa_per_row = [[] for _ in range(len(dataset))]

    for call_idx, raw_response in enumerate(model_responses):
        row_idx = call_index_to_row_id[call_idx]
        # We'll parse the response to extract <output_json> block
        extracted_json_str = _extract_tag_content(raw_response, "output_json")
        if not extracted_json_str.strip():
            logger.warning("No <output_json> block found for row {}. Skipping multi-hop output.", row_idx)
            continue

        try:
            question_answer_pairs = json.loads(extracted_json_str)
            # Parse each object in the question_answer_pairs list
            for qap in question_answer_pairs:
                # We don't have chunk-level IDs in multi-hop, but let's define chunk_id = row_idx, or  row_idx + something
                # or simply "multihop_{row_idx}". For more fine-grained, you might store them if there's multiple multi-hop chunks.
                chunk_id = f"multihop_{row_idx}"
                document_id = dataset[row_idx].get("document_id", f"doc_{row_idx}")
                question = qap.get("question", "")
                answer = qap.get("answer", "")
                difficulty = qap.get("estimated_difficulty", 5)
                question_type = qap.get("question_type", "unknown")
                self_answer = qap.get("thought_process", "")

                multi_hop_q = MultiHopQuestion(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    question=question,
                    answer=answer,
                    self_estimated_difficulty_rating=difficulty,
                    self_assessed_question_type=question_type,
                    self_answer=self_answer,
                )
                multi_hop_qa_per_row[row_idx].append({
                    "chunk_id": multi_hop_q.chunk_id,
                    "document_id": multi_hop_q.document_id,
                    "question": multi_hop_q.question,
                    "answer": multi_hop_q.answer,
                    "self_estimated_difficulty_rating": multi_hop_q.self_estimated_difficulty_rating,
                    "self_assessed_question_type": multi_hop_q.self_assessed_question_type,
                    "self_answer": multi_hop_q.self_answer,
                })
        except Exception as e:
            logger.warning("Failed to parse JSON for row {}: {}", row_idx, e)

    # === Add the new column and save ===
    dataset = dataset.add_column("multi_hop_questions", multi_hop_qa_per_row)
    dataset = dataset.add_column("multi_hop_generating_model", [generating_model_name] * len(dataset))

    logger.info("Saving multi-hop questions dataset to {}.", output_dataset_name)
    save_dataset(
        dataset=dataset,
        step_name="multi_hop_question_generation",
        config=config,
        output_dataset_name=output_dataset_name
    )
    logger.success("Multi-hop question generation complete.")


def _extract_tag_content(text: str, tag: str) -> str:
    """
    Helper function: Extract content enclosed in <tag>...</tag> from a string (non-greedy).
    Returns the first match or empty string if none found.
    """
    import re
    pattern = fr"<{tag}\s*>([\s\S]*?)</{tag}>"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return ""
