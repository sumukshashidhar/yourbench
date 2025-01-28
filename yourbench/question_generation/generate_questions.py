import ast
import uuid
from typing import List

from datasets import Dataset, load_dataset
from models.single_shot_question import (
    QuestionAnswerPair,
    QuestionAnswerPairWithThoughtProcess,
)
from utils.dataset_engine import handle_dataset_push, make_dataset_name
from utils.inference_engine import run_parallel_inference
from utils.load_prompt import load_prompt
from utils.parsing_engine import extract_content_from_xml_tags


def _clean_questions(text: str):
    text = text.replace("```json", "").replace("```", "")
    try:
        cleaned = ast.literal_eval(text)
        if not isinstance(cleaned, list):
            return []
        return cleaned
    except Exception as _:
        return []


def _validate_questions_multihop(questions: List[QuestionAnswerPairWithThoughtProcess]):
    # check if the questions are valid, with the pydantic model
    validated_questions = []
    for i in range(len(questions)):
        try:
            validated_questions.append(QuestionAnswerPairWithThoughtProcess(**questions[i]).model_dump())
        except Exception as _:
            continue
    return validated_questions


def _validate_questions(questions: List[QuestionAnswerPair]):
    # check if the questions are valid, with the pydantic model
    validated_questions = []
    for i in range(len(questions)):
        try:
            validated_questions.append(QuestionAnswerPairWithThoughtProcess(**questions[i]).model_dump())
        except Exception as _:
            continue
    return validated_questions


def generate_multihop_questions(config: dict):
    # load the multi-hop pairings dataset
    multi_hop_chunks_dataset_name = config["pipeline"]["create_multi_hop_questions"]["source_dataset_name"]
    multi_hop_questions_dataset_name = config["pipeline"]["create_multi_hop_questions"]["target_dataset_name"]

    test_audience = config["pipeline"]["create_multi_hop_questions"]["test_audience"]

    # load the dataset
    multi_hop_chunks_dataset = load_dataset(make_dataset_name(config, multi_hop_chunks_dataset_name), split="train")

    # load the prompts
    system_prompt = load_prompt(
        f"{config['pipeline']['create_multi_hop_questions']['prompt_prefix']}.fast_multi_hop_system"
    )
    user_prompt = load_prompt(
        f"{config['pipeline']['create_multi_hop_questions']['prompt_prefix']}.fast_multi_hop_user"
    )

    # create the prompts to batch with
    prompts = []
    for multihop_pairing in multi_hop_chunks_dataset:
        multihop_pairing["document_id"]
        document_name = multihop_pairing["document_name"]
        document_summary = multihop_pairing["document_summary"]
        chunk_ids = multihop_pairing["chunk_ids"]
        len(chunk_ids)
        text_chunks = multihop_pairing["chunks"]

        # format the chunks properly
        chunks = ""
        for i, chunk in enumerate(text_chunks):
            chunks += f"<text_chunk_{i}>\n{chunk}\n</text_chunk_{i}>\n"

        prompt = user_prompt.format(
            title=document_name,
            document_summary=document_summary,
            chunks=chunks,
            test_audience=test_audience,
        )

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        prompts.append(message)

    # do inference on the messages.
    responses = run_parallel_inference(prompts=prompts, config=config)
    # now, extract the questions properly, as well as the document analysis
    document_analysis = [extract_content_from_xml_tags(response, "document_analysis") for response in responses]
    questions = [extract_content_from_xml_tags(response, "output_json") for response in responses]

    new_dataset_rows = []
    for i in range(len(multi_hop_chunks_dataset)):
        new_dataset_rows.append({
            # from multihop chunks dataset
            "document_id": multi_hop_chunks_dataset[i]["document_id"],
            "document_name": multi_hop_chunks_dataset[i]["document_name"],
            "document_summary": multi_hop_chunks_dataset[i]["document_summary"],
            "document_category": multi_hop_chunks_dataset[i]["document_category"],
            "chunk_ids": multi_hop_chunks_dataset[i]["chunk_ids"],
            "chunks": multi_hop_chunks_dataset[i]["chunks"],
            # from inference
            "document_analysis": document_analysis[i],
            "test_audience": test_audience,
            "questions": questions[i],
        })
    new_dataset_rows_expanded = []
    for i in range(len(new_dataset_rows)):
        cleaned_questions = _clean_questions(new_dataset_rows[i]["questions"])
        validated_questions = _validate_questions_multihop(cleaned_questions)
        for question in validated_questions:
            new_dataset_rows_expanded.append({
                # document specific info
                "document_id": new_dataset_rows[i]["document_id"],
                "document_name": new_dataset_rows[i]["document_name"],
                "document_summary": new_dataset_rows[i]["document_summary"],
                "document_category": new_dataset_rows[i]["document_category"],
                "chunk_ids": new_dataset_rows[i]["chunk_ids"],
                "chunks": new_dataset_rows[i]["chunks"],
                # question specific info
                "question_id": str(uuid.uuid4()),
                "test_audience": new_dataset_rows[i]["test_audience"],
                "document_analysis": new_dataset_rows[i]["document_analysis"],
                "question_type": question["question_type"],
                "thought_process": question["thought_process"],
                "question": question["question"],
                "answer": question["answer"],
                "estimated_difficulty": question["estimated_difficulty"],
                "citations": str(question["citations"]),
                "generating_model": config["configurations"]["model"]["model_name"],
            })
    new_dataset = Dataset.from_list(new_dataset_rows_expanded)
    handle_dataset_push(
        config,
        multi_hop_questions_dataset_name,
        new_dataset,
    )


def generate_single_shot_questions(config: dict):
    # load the chunk dataset
    chunk_dataset_name = config["pipeline"]["create_single_hop_questions"]["source_dataset_name"]
    chunk_dataset = load_dataset(make_dataset_name(config, chunk_dataset_name), split="train")
    # load the prompt
    system_prompt = load_prompt(
        f"{config['pipeline']['create_single_hop_questions']['prompt_prefix']}.fast_single_shot_system"
    )
    user_prompt = load_prompt(
        f"{config['pipeline']['create_single_hop_questions']['prompt_prefix']}.fast_single_shot_user"
    )

    # create the prompts to batch with
    prompts = []
    for chunk_row in chunk_dataset:
        title = chunk_row["document_name"]
        document_summary = chunk_row["document_summary"]
        text_chunk = chunk_row["chunk"]
        test_audience = config["pipeline"]["create_single_hop_questions"]["test_audience"]

        prompt = user_prompt.format(
            title=title,
            document_summary=document_summary,
            text_chunk=text_chunk,
            test_audience=test_audience,
        )

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompts.append(message)

    # do inference on the messages.
    responses = run_parallel_inference(prompts=prompts, config=config)
    # now, extract the questions properly, as well as the document analysis
    document_analysis = [extract_content_from_xml_tags(response, "document_analysis") for response in responses]
    questions = [extract_content_from_xml_tags(response, "output_json") for response in responses]

    # make a new huggingface dataset
    new_dataset_rows = []
    for i in range(len(chunk_dataset)):
        new_dataset_rows.append({
            "document_id": chunk_dataset[i]["document_id"],
            "document_name": chunk_dataset[i]["document_name"],
            "document_summary": chunk_dataset[i]["document_summary"],
            "document_category": chunk_dataset[i]["document_category"],
            "chunk_location_id": chunk_dataset[i]["chunk_location_id"],
            "chunk": chunk_dataset[i]["chunk"],
            "test_audience": config["pipeline"]["create_single_hop_questions"]["test_audience"],
            "full_generation_response": responses[i],
            "document_analysis": document_analysis[i],
            "questions": questions[i],
        })
    # okay, now, we need to expand each of the questions and parse the list
    new_dataset_rows_expanded = []
    for i in range(len(new_dataset_rows)):
        cleaned_questions = _clean_questions(new_dataset_rows[i]["questions"])
        validated_questions = _validate_questions(cleaned_questions)
        for question in validated_questions:
            new_dataset_rows_expanded.append({
                "question_id": str(uuid.uuid4()),
                "document_id": new_dataset_rows[i]["document_id"],
                "document_name": new_dataset_rows[i]["document_name"],
                "document_category": new_dataset_rows[i]["document_category"],
                "document_summary": new_dataset_rows[i]["document_summary"],
                "chunk_location_id": new_dataset_rows[i]["chunk_location_id"],
                "chunk": new_dataset_rows[i]["chunk"],
                "test_audience": new_dataset_rows[i]["test_audience"],
                "full_generation_response": new_dataset_rows[i]["full_generation_response"],
                "document_analysis": new_dataset_rows[i]["document_analysis"],
                "question_type": question["question_type"],
                "thought_process": question["thought_process"],
                "question": question["question"],
                "answer": question["answer"],
                "estimated_difficulty": question["estimated_difficulty"],
                "citations": str(question["citations"]),
                "generating_model": config["configurations"]["model"]["model_name"],
            })
    new_dataset = Dataset.from_list(new_dataset_rows_expanded)
    handle_dataset_push(
        config,
        config["pipeline"]["create_single_hop_questions"]["target_dataset_name"],
        new_dataset,
    )
