from datasets import Dataset, concatenate_datasets, load_dataset
from loguru import logger

from yourbench.utils.inference_engine import run_parallel_inference
from yourbench.utils.load_prompt import load_prompt
from yourbench.utils.parsing_engine import extract_content_from_xml_tags


def handle_dataset_push(dataset: Dataset, dataset_name: str, config: dict) -> None:
    if config["configurations"]["push_to_huggingface"]:
        privacy = False if config["configurations"]["set_hf_repo_visibility"] != "private" else True
        logger.info(f"Pushing dataset '{dataset_name}' to Hugging Face Hub (privacy={privacy})")

        try:
            hub_path = f"{config['configurations']['hf_organization']}/{dataset_name}"
            # Try to load existing dataset
            try:
                existing_dataset = load_dataset(hub_path, split="train")
                logger.info(f"Found existing dataset at {hub_path}, concatenating...")
                dataset = concatenate_datasets([existing_dataset, dataset])
            except Exception as _:
                logger.info(f"No existing dataset found at {hub_path}, creating new...")

            # Push the dataset (either concatenated or new)
            dataset.push_to_hub(hub_path, private=privacy)
            logger.success(f"Successfully pushed dataset to Hugging Face Hub: {dataset_name}")
        except Exception as error:
            logger.error(f"Failed to push dataset to Hugging Face Hub: {str(error)}")
            raise
    else:
        logger.info(f"Saving dataset locally to: {dataset_name}")
        dataset.save_to_disk(dataset_name)
        logger.success(f"Successfully saved dataset to disk: {dataset_name}")


def _get_full_dataset_name_for_questions(config: dict) -> str:
    return config["configurations"]["hf_organization"] + "/" + config["selected_choices"]["answer_questions_with_llm"]["source_dataset_name"]


def _get_zeroshot_answers(config: dict) -> dict:
    # load the dataset
    dataset = load_dataset(_get_full_dataset_name_for_questions(config))
    dataset = dataset["train"]
    # dataset = dataset.select(range(10))

    # extract the questions
    questions = dataset["question"]
    # pass the relevant information to the prompt
    prompt = load_prompt(f'{config["selected_choices"]["answer_questions_with_llm"]["prompt_prefix"]}.fast_answer_q_zeroshot_user')
    prompts = [prompt.format(question=question) for question in questions]

    # create the messages
    messages = []
    for prompt in prompts:
        messages.append([{"role" : "user", "content" : prompt}])


    # get the responses
    responses = run_parallel_inference(messages, config)
    # now, we need to save the responses to a new dataset
    # extract the answer from the response from the xml tags
    answers = [extract_content_from_xml_tags(response, "answer") for response in responses]

    dataset_rows = []
    for i, (question, answer) in enumerate(zip(questions, answers)):
        oracle = dataset["answer"][i]
        # print(oracle)
        dataset_rows.append({
            "title": dataset["title"][i],
            "summary": dataset["summary"][i],
            "chunk": dataset["chunk"][i],
            "test_audience": dataset["test_audience"][i],
            "question_id": dataset["question_id"][i],
            "question_type": dataset["question_type"][i],
            "estimated_difficulty": dataset["estimated_difficulty"][i],
            "citations": dataset["citations"][i],
            "question": question,
            "oracle_answer": oracle,
            "full_response": responses[i],
            "answer": answer,
            "scenario": "zero_shot",
            "generating_model": config["configurations"]["model"]["model_name"]
        })
    # Create the dataset
    new_dataset = Dataset.from_list(dataset_rows)

    # Get the dataset name from config
    dataset_name = config["selected_choices"]["answer_questions_with_llm"]["answer_scenarios"]["zero_shot"]["answer_dataset_name"]

    # Handle dataset push (reusing the function from generate_questions.py)
    handle_dataset_push(new_dataset, dataset_name, config)

def _get_relevant_chunk_answers(config: dict):
    # load the dataset
    dataset = load_dataset(_get_full_dataset_name_for_questions(config))
    # extract the questions
    questions = dataset["train"]["question"]
    chunks = dataset["train"]["chunk"]
    # pass the relevant information to the prompt
    prompt = load_prompt(f'{config["selected_choices"]["answer_questions_with_llm"]["prompt_prefix"]}.fast_answer_q_relevant_chunk_user')
    prompts = [prompt.format(question=question, document=document) for question, document in zip(questions, chunks)]

    # create the messages
    messages = []
    for prompt in prompts:
        messages.append([{"role" : "user", "content" : prompt}])


    # get the responses
    responses = run_parallel_inference(messages, config)
    # now, we need to save the responses to a new dataset
    # extract the answer from the response from the xml tags
    answers = [extract_content_from_xml_tags(response, "answer") for response in responses]

    dataset_rows = []
    for i, (question, answer) in enumerate(zip(questions, answers)):
        dataset_rows.append({
            "title": dataset["train"]["title"][i],
            "summary": dataset["train"]["summary"][i],
            "chunk": dataset["train"]["chunk"][i],
            "test_audience": dataset["train"]["test_audience"][i],
            "question_id": dataset["train"]["question_id"][i],
            "question_type": dataset["train"]["question_type"][i],
            "estimated_difficulty": dataset["train"]["estimated_difficulty"][i],
            "citations": dataset["train"]["citations"][i],
            "question": question,
            "oracle_answer": dataset["train"]["answer"][i],
            "full_response": responses[i],
            "answer": answer,
            "scenario": "answer_with_relevant_chunks",
            "generating_model": config["configurations"]["model"]["model_name"]
        })
    # Create the dataset
    new_dataset = Dataset.from_list(dataset_rows)

    # Get the dataset name from config
    dataset_name = config["selected_choices"]["answer_questions_with_llm"]["answer_scenarios"]["answer_with_relevant_chunks"]["answer_dataset_name"]

    # Handle dataset push (reusing the function from generate_questions.py)
    handle_dataset_push(new_dataset, dataset_name, config)

def _get_zeroshot_cot_answers(config: dict):
    # load the dataset
    dataset = load_dataset(_get_full_dataset_name_for_questions(config))
    # extract the questions
    questions = dataset["train"]["question"]
    chunks = dataset["train"]["chunk"]
    # pass the relevant information to the prompt
    prompt = load_prompt(f'{config["selected_choices"]["answer_questions_with_llm"]["prompt_prefix"]}.fast_answer_q_cot_user')
    prompts = [prompt.format(question=question, document=document) for question, document in zip(questions, chunks)]

    # create the messages
    messages = []
    for prompt in prompts:
        messages.append([{"role" : "user", "content" : prompt}])


    # get the responses
    responses = run_parallel_inference(messages, config)
    # now, we need to save the responses to a new dataset
    # extract the answer from the response from the xml tags
    answers = [extract_content_from_xml_tags(response, "answer") for response in responses]

    dataset_rows = []
    for i, (question, answer) in enumerate(zip(questions, answers)):
        dataset_rows.append({
            "title": dataset["train"]["title"][i],
            "summary": dataset["train"]["summary"][i],
            "chunk": dataset["train"]["chunk"][i],
            "test_audience": dataset["train"]["test_audience"][i],
            "question_id": dataset["train"]["question_id"][i],
            "question_type": dataset["train"]["question_type"][i],
            "estimated_difficulty": dataset["train"]["estimated_difficulty"][i],
            "citations": dataset["train"]["citations"][i],
            "question": question,
            "oracle_answer": dataset["train"]["answer"][i],
            "full_response": responses[i],
            "answer": answer,
            "scenario": "zero_shot_with_cot",
            "generating_model": config["configurations"]["model"]["model_name"]
        })
    # Create the dataset
    new_dataset = Dataset.from_list(dataset_rows)

    # Get the dataset name from config
    dataset_name = config["selected_choices"]["answer_questions_with_llm"]["answer_scenarios"]["zero_shot_with_cot"]["answer_dataset_name"]

    # Handle dataset push (reusing the function from generate_questions.py)
    handle_dataset_push(new_dataset, dataset_name, config)

def _get_document_summary_answers(config: dict):
    # load the dataset
    dataset = load_dataset(_get_full_dataset_name_for_questions(config))
    # extract the questions
    questions = dataset["train"]["question"]
    chunks = dataset["train"]["chunk"]
    summaries = dataset["train"]["summary"]
    # pass the relevant information to the prompt
    prompt = load_prompt(f'{config["selected_choices"]["answer_questions_with_llm"]["prompt_prefix"]}.fast_answer_q_docsummary_user')
    prompts = [prompt.format(question=question, summary=summary) for question, summary in zip(questions, summaries)]

    # create the messages
    messages = []
    for prompt in prompts:
        messages.append([{"role" : "user", "content" : prompt}])


    # get the responses
    responses = run_parallel_inference(messages, config)
    # now, we need to save the responses to a new dataset
    # extract the answer from the response from the xml tags
    answers = [extract_content_from_xml_tags(response, "answer") for response in responses]

    dataset_rows = []
    for i, (question, answer) in enumerate(zip(questions, answers)):
        dataset_rows.append({
            "title": dataset["train"]["title"][i],
            "summary": dataset["train"]["summary"][i],
            "chunk": dataset["train"]["chunk"][i],
            "test_audience": dataset["train"]["test_audience"][i],
            "question_id": dataset["train"]["question_id"][i],
            "question_type": dataset["train"]["question_type"][i],
            "estimated_difficulty": dataset["train"]["estimated_difficulty"][i],
            "citations": dataset["train"]["citations"][i],
            "question": question,
            "oracle_answer": dataset["train"]["answer"][i],
            "full_response": responses[i],
            "answer": answer,
            "scenario": "answer_with_document_summary",
            "generating_model": config["configurations"]["model"]["model_name"]
        })
    # Create the dataset
    new_dataset = Dataset.from_list(dataset_rows)

    # Get the dataset name from config
    dataset_name = config["selected_choices"]["answer_questions_with_llm"]["answer_scenarios"]["answer_with_document_summary"]["answer_dataset_name"]

    # Handle dataset push (reusing the function from generate_questions.py)
    handle_dataset_push(new_dataset, dataset_name, config)

def _get_gold_answers(config: dict):
    # load the dataset
    dataset = load_dataset(_get_full_dataset_name_for_questions(config))
    dataset = dataset["train"]
    # extract the questions
    questions = dataset["question"]
    chunks = dataset["chunk"]
    summaries = dataset["summary"]
    # pass the relevant information to the prompt
    prompt = load_prompt(f'{config["selected_choices"]["answer_questions_with_llm"]["prompt_prefix"]}.fast_answer_q_gold_user')
    prompts = [prompt.format(question=question, document=document, summary=summary) for question, document, summary in zip(questions, chunks, summaries)]

    # create the messages
    messages = []
    for prompt in prompts:
        messages.append([{"role" : "user", "content" : prompt}])


    # get the responses
    responses = run_parallel_inference(messages, config)
    # now, we need to save the responses to a new dataset
    # extract the answer from the response from the xml tags
    answers = [extract_content_from_xml_tags(response, "answer") for response in responses]

    dataset_rows = []
    for i, (question, answer) in enumerate(zip(questions, answers)):
        oracle = dataset["answer"][i]
        dataset_rows.append({
            "title": dataset["title"][i],
            "summary": dataset["summary"][i],
            "chunk": dataset["chunk"][i],
            "test_audience": dataset["test_audience"][i],
            "question_id": dataset["question_id"][i],
            "question_type": dataset["question_type"][i],
            "estimated_difficulty": dataset["estimated_difficulty"][i],
            "citations": dataset["citations"][i],
            "question": question,
            "oracle_answer": oracle,
            "full_response": responses[i],
            "answer": answer,
            "scenario": "gold_standard",
            "generating_model": config["configurations"]["model"]["model_name"]
        })
    # Create the dataset
    new_dataset = Dataset.from_list(dataset_rows)

    # Get the dataset name from config
    dataset_name = config["selected_choices"]["answer_questions_with_llm"]["answer_scenarios"]["gold_standard"]["answer_dataset_name"]

    # Handle dataset push (reusing the function from generate_questions.py)
    handle_dataset_push(new_dataset, dataset_name, config)



def answer_questions_with_llm(config: dict):
    answer_scenarios = config["selected_choices"]["answer_questions_with_llm"]["answer_scenarios"]
    if answer_scenarios["zero_shot"]["execute"]:
        _get_zeroshot_answers(config)
    if answer_scenarios["zero_shot_with_cot"]["execute"]:
        _get_zeroshot_cot_answers(config)
    if answer_scenarios["answer_with_document_summary"]["execute"]:
        _get_document_summary_answers(config)
    if answer_scenarios["answer_with_relevant_chunks"]["execute"]:
        _get_relevant_chunk_answers(config)
    if answer_scenarios["gold_standard"]["execute"]:
        _get_gold_answers(config)

    pass
