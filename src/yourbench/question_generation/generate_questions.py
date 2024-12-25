import ast
import uuid
from typing import Dict, List

from datasets import Dataset, concatenate_datasets, load_dataset
from loguru import logger

from yourbench.models.single_shot_question import (
    QuestionAnswerPair,
    QuestionAnswerPairWithThoughtProcess,
)
from yourbench.utils.inference_engine import run_parallel_inference
from yourbench.utils.load_prompt import load_prompt
from yourbench.utils.parsing_engine import extract_content_from_xml_tags
from yourbench.utils.dataset_engine import handle_dataset_push, make_dataset_name


def get_full_dataset_name_for_multihop_questions(config: Dict) -> str:
    source_dataset_name = config["selected_choices"]["create_multihop_questions"][
        "source_dataset_name"
    ]
    return config["configurations"]["hf_organization"] + "/" + source_dataset_name


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
            validated_questions.append(
                QuestionAnswerPairWithThoughtProcess(**questions[i]).model_dump()
            )
        except Exception as _:
            continue
    return validated_questions


def _validate_questions(questions: List[QuestionAnswerPair]):
    # check if the questions are valid, with the pydantic model
    validated_questions = []
    for i in range(len(questions)):
        try:
            validated_questions.append(
                QuestionAnswerPairWithThoughtProcess(**questions[i]).model_dump()
            )
        except Exception as _:
            continue
    return validated_questions


def generate_multihop_questions(config: dict):
    multihop_pairings = load_dataset(
        get_full_dataset_name_for_multihop_questions(config), split="train"
    )
    # load the prompt
    system_prompt = load_prompt(
        f'{config["selected_choices"]["create_multihop_questions"]["prompt_prefix"]}.fast_multi_hop_system'
    )
    user_prompt = load_prompt(
        f'{config["selected_choices"]["create_multihop_questions"]["prompt_prefix"]}.fast_multi_hop_user'
    )
    # create the prompts to batch with
    prompts = []
    for multihop_pairing in multihop_pairings:
        title = multihop_pairing["title"]
        document_summary = multihop_pairing["summary"]
        test_audience = config["selected_choices"]["create_multihop_questions"][
            "test_audience"
        ]
        text_chunks = multihop_pairing["chunks"]

        # format the chunks properly
        chunks = ""
        for i, chunk in enumerate(text_chunks):
            chunks += f"<text_chunk_{i}>\n{chunk}\n</text_chunk_{i}>\n"

        prompt = user_prompt.format(
            title=title,
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
    document_analysis = [
        extract_content_from_xml_tags(response, "document_analysis")
        for response in responses
    ]
    questions = [
        extract_content_from_xml_tags(response, "output_json") for response in responses
    ]

    # make a new huggingface dataset
    new_dataset_rows = []
    for i in range(len(multihop_pairings)):
        new_dataset_rows.append(
            {
                "title": multihop_pairings[i]["title"],
                "summary": multihop_pairings[i]["summary"],
                "chunk_ids": multihop_pairings[i]["chunk_ids"],
                "chunks": multihop_pairings[i]["chunks"],
                "document_analysis": document_analysis[i],
                "test_audience": config["selected_choices"][
                    "create_multihop_questions"
                ]["test_audience"],
                "questions": questions[i],
            }
        )

    # okay, now, we need to expand each of the questions and parse the list
    new_dataset_rows_expanded = []
    for i in range(len(new_dataset_rows)):
        cleaned_questions = _clean_questions(new_dataset_rows[i]["questions"])
        validated_questions = _validate_questions_multihop(cleaned_questions)
        for question in validated_questions:
            new_dataset_rows_expanded.append(
                {
                    "question_id": str(uuid.uuid4()),
                    "title": new_dataset_rows[i]["title"],
                    "summary": new_dataset_rows[i]["summary"],
                    "chunk_ids": new_dataset_rows[i]["chunk_ids"],
                    "chunks": new_dataset_rows[i]["chunks"],
                    "test_audience": new_dataset_rows[i]["test_audience"],
                    "document_analysis": new_dataset_rows[i]["document_analysis"],
                    "question_type": question["question_type"],
                    "thought_process": question["thought_process"],
                    "question": question["question"],
                    "answer": question["answer"],
                    "estimated_difficulty": question["estimated_difficulty"],
                    "citations": str(question["citations"]),
                    "generating_model": config["configurations"]["model"]["model_name"],
                }
            )

    new_dataset = Dataset.from_list(new_dataset_rows_expanded)
    handle_dataset_push(
        new_dataset,
        config["selected_choices"]["create_multihop_questions"][
            "multihop_questions_dataset_name"
        ],
        config,
    )


def generate_single_shot_questions(config: dict):
    # load the chunk dataset
    chunk_dataset_name = config["pipeline"]["create_single_hop_questions"][
        "source_dataset_name"
    ]
    chunk_dataset = load_dataset(
        make_dataset_name(config, chunk_dataset_name), split="train"
    )
    # load the prompt
    system_prompt = load_prompt(
        f'{config["pipeline"]["create_single_hop_questions"]["prompt_prefix"]}.fast_single_shot_system'
    )
    user_prompt = load_prompt(
        f'{config["pipeline"]["create_single_hop_questions"]["prompt_prefix"]}.fast_single_shot_user'
    )

    # create the prompts to batch with
    prompts = []
    for chunk_row in chunk_dataset:
        title = chunk_row["document_name"]
        document_summary = chunk_row["document_summary"]
        text_chunk = chunk_row["chunk"]
        test_audience = config["pipeline"]["create_single_hop_questions"][
            "test_audience"
        ]

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
    document_analysis = [
        extract_content_from_xml_tags(response, "document_analysis")
        for response in responses
    ]
    questions = [
        extract_content_from_xml_tags(response, "output_json") for response in responses
    ]

    # make a new huggingface dataset
    new_dataset_rows = []
    for i in range(len(chunk_dataset)):
        new_dataset_rows.append(
            {
                "document_id": chunk_dataset[i]["document_id"],
                "document_name": chunk_dataset[i]["document_name"],
                "document_summary": chunk_dataset[i]["document_summary"],
                "chunk_location_id": chunk_dataset[i]["chunk_location_id"],
                "chunk": chunk_dataset[i]["chunk"],
                "test_audience": config["pipeline"]["create_single_hop_questions"][
                    "test_audience"
                ],
                "full_generation_response": responses[i],
                "document_analysis": document_analysis[i],
                "questions": questions[i],
            }
        )
    # okay, now, we need to expand each of the questions and parse the list
    new_dataset_rows_expanded = []
    for i in range(len(new_dataset_rows)):
        cleaned_questions = _clean_questions(new_dataset_rows[i]["questions"])
        validated_questions = _validate_questions(cleaned_questions)
        for question in validated_questions:
            new_dataset_rows_expanded.append(
                {
                    "question_id": str(uuid.uuid4()),
                    "document_id": new_dataset_rows[i]["document_id"],
                    "document_name": new_dataset_rows[i]["document_name"],
                    "document_summary": new_dataset_rows[i]["document_summary"],
                    "chunk_location_id": new_dataset_rows[i]["chunk_location_id"],
                    "chunk": new_dataset_rows[i]["chunk"],
                    "test_audience": new_dataset_rows[i]["test_audience"],
                    "full_generation_response": new_dataset_rows[i][
                        "full_generation_response"
                    ],
                    "document_analysis": new_dataset_rows[i]["document_analysis"],
                    "question_type": question["question_type"],
                    "thought_process": question["thought_process"],
                    "question": question["question"],
                    "answer": question["answer"],
                    "estimated_difficulty": question["estimated_difficulty"],
                    "citations": str(question["citations"]),
                    "generating_model": config["configurations"]["model"]["model_name"],
                }
            )
    new_dataset = Dataset.from_list(new_dataset_rows_expanded)
    handle_dataset_push(
        config,
        config["pipeline"]["create_single_hop_questions"]["target_dataset_name"],
        new_dataset,
    )
