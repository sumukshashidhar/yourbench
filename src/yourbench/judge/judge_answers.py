from datasets import Dataset, concatenate_datasets, load_dataset
from loguru import logger

from yourbench.utils.inference_engine import run_parallel_inference
from yourbench.utils.load_prompt import load_prompt
from yourbench.utils.parsing_engine import extract_content_from_xml_tags
from yourbench.utils.dataset_engine import make_dataset_name, handle_dataset_push


def judge_answers(config: dict):
    """
    Judge the answers
    """
    try:
        # Load dataset
        dataset_name = make_dataset_name(
            config, config["pipeline"]["judge"]["source_dataset_name"]
        )
        dataset = load_dataset(dataset_name, split="train")

        print(dataset.column_names)

        # Validate required columns
        required_columns = [
            "question",
            "oracle_answer",
            "chunk",
            "document_summary",
            "answer_a",
            "answer_b",
        ]
        missing_columns = [
            col for col in required_columns if col not in dataset.column_names
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns in dataset: {missing_columns}")

        # Load prompts
        prompt_base = config["pipeline"]["judge"]["prompt_prefix"]
        prompt_name = config["pipeline"]["judge"]["judge_prompt_name"]
        try:
            prompt_system = load_prompt(f"{prompt_base}.{prompt_name}_system")
            prompt_user = load_prompt(f"{prompt_base}.{prompt_name}_user")
        except Exception as e:
            raise ValueError(f"Failed to load prompts: {str(e)}")

        # prompts are loaded, dataset is loaded, now we need to judge the answers.
        questions = dataset["question"]
        oracle_answers = dataset["oracle_answer"]
        chunks = dataset["chunk"]
        summaries = dataset["document_summary"]
        answers_a = dataset["answer_a"]
        answers_b = dataset["answer_b"]

        # make prompts
        prompts = []
        for i in range(len(questions)):
            prompt = prompt_user.format(
                question=questions[i],
                oracle_answer=oracle_answers[i],
                chunk=chunks[i],
                summary=summaries[i],
                answer_a=answers_a[i],
                answer_b=answers_b[i],
            )
            prompts.append(prompt)
        messages = []
        for prompt in prompts:
            messages.append(
                [
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt},
                ]
            )
        results = run_parallel_inference(messages, config)
        final_answers = [
            extract_content_from_xml_tags(result, "final_answer") for result in results
        ]
        # now, we need to save the results
        # add new columns to the dataset
        dataset = dataset.add_column("judge_full_result", results)
        dataset = dataset.add_column("evaluation_result", final_answers)
        dataset = dataset.add_column(
            "judge_model_name",
            [config["configurations"]["model"]["model_name"]] * len(questions),
        )
        # now, we need to save the dataset
        handle_dataset_push(
            config,
            make_dataset_name(
                config, config["pipeline"]["judge"]["target_dataset_name"]
            ),
            dataset,
        )
    except Exception as e:
        logger.error(f"Failed to judge answers: {str(e)}")
        raise
