import ast
from typing import List

from datasets import Dataset, concatenate_datasets, load_dataset

from yourbench.models.single_shot_question import QuestionAnswerPair
from yourbench.utils.inference_engine import run_parallel_inference
from yourbench.utils.load_prompt import load_prompt
from yourbench.utils.parsing_engine import extract_content_from_xml_tags

MODEL_INDEX = 1

def _clean_questions(text: str):
    text = text.replace("```json", "").replace("```", "")
    try:
        cleaned = ast.literal_eval(text)
        if not isinstance(cleaned, list):
            return []
        return cleaned
    except Exception as _:
        return []


def _validate_questions(questions: List[QuestionAnswerPair]):
    # check if the questions are valid, with the pydantic model
    validated_questions = []
    for i in range(len(questions)):
        try:
            validated_questions.append(
                QuestionAnswerPair(**questions[i]).model_dump()
                )
        except Exception as _:
            continue
    return validated_questions


def generate_single_shot_questions(document_dataset_name: str, config: dict):
    # load the chunk dataset
    chunk_dataset = load_dataset(config["datasets"]["chunked_doucments_dataset_name"], split="train")
    # load the prompt
    system_prompt = load_prompt(f'{config["question_generation_config"]["prompt_prefix"]}.fast_single_shot_system')
    user_prompt = load_prompt(f'{config["question_generation_config"]["prompt_prefix"]}.fast_single_shot_user')

    # create the prompts to batch with
    prompts = []
    for chunk_row in chunk_dataset:
        title = chunk_row["title"]
        document_summary = chunk_row["summary"]
        text_chunk = chunk_row["chunk"]
        test_audience = config["question_generation_config"]["test_audience"]

        prompt = user_prompt.format(title=title, document_summary=document_summary, text_chunk=text_chunk, test_audience=test_audience)

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        prompts.append(message)

    # do inference on the messages.
    responses = run_parallel_inference(model_selection=MODEL_INDEX, prompts=prompts, config=config)
    # now, extract the questions properly, as well as the document analysis
    document_analysis = [extract_content_from_xml_tags(response, "document_analysis") for response in responses]
    questions = [extract_content_from_xml_tags(response, "output_json") for response in responses]

    # make a new huggingface dataset
    new_dataset_rows = []
    for i in range(len(chunk_dataset)):
        new_dataset_rows.append({
            "title": chunk_dataset[i]["title"],
            "summary": chunk_dataset[i]["summary"],
            "chunk": chunk_dataset[i]["chunk"],
            "test_audience": config["question_generation_config"]["test_audience"],
            "document_analysis": document_analysis[i],
            "questions": questions[i]
        })
    # okay, now, we need to expand each of the questions and parse the list
    new_dataset_rows_expanded = []
    for i in range(len(new_dataset_rows)):
        cleaned_questions = _clean_questions(new_dataset_rows[i]["questions"])
        validated_questions = _validate_questions(cleaned_questions)
        for question in validated_questions:
            new_dataset_rows_expanded.append({
                "title": new_dataset_rows[i]["title"],
                "summary": new_dataset_rows[i]["summary"],
                "chunk": new_dataset_rows[i]["chunk"],
                "test_audience": new_dataset_rows[i]["test_audience"],
                "document_analysis": new_dataset_rows[i]["document_analysis"],
                "question_type": question["question_type"],
                "question": question["question"],
                "answer": question["answer"],
                "estimated_difficulty": question["estimated_difficulty"],
                "citations": str(question["citations"]),
                "generating_model": config["model_config"][f"model_{MODEL_INDEX}"]["model_name"]
            })
    new_dataset = Dataset.from_list(new_dataset_rows_expanded)

    # check if the single_shot_questions_dataset_name exists on the hub
    try:
        existing_dataset = load_dataset(config["datasets"]["single_shot_questions_dataset_name"], split="train")
        # it exists, so we need to append to the dataset
        new_dataset = concatenate_datasets([existing_dataset, new_dataset])
        new_dataset.push_to_hub(config["datasets"]["single_shot_questions_dataset_name"], private=True)
    except Exception as _:
        # it does not exist, so we need to create a new dataset
        new_dataset.push_to_hub(config["datasets"]["single_shot_questions_dataset_name"], private=True)

    return responses
