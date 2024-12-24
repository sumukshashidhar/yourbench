from datasets import load_dataset

from yourbench.utils.inference_engine import run_parallel_inference
from yourbench.utils.load_prompt import load_prompt
from yourbench.utils.load_task_config import _get_full_dataset_name_for_questions


def judge_answers(config: dict):
    """
    Judge the answers
    """
    # read the dataset
    dataset = load_dataset(_get_full_dataset_name_for_questions(config, config["selected_choices"]["judge"]["source_dataset_name"]))
    # now, we have the source dataset. let's load the prompt
    prompt_system = config["selected_choices"]["judge"]["prompt_prefix"] + "." + config["selected_choices"]["judge"]["judge_prompt_name"] + "_system"
    prompt_user = config["selected_choices"]["judge"]["prompt_prefix"] + "." + config["selected_choices"]["judge"]["judge_prompt_name"] + "_user"
    prompt_system = load_prompt(prompt_system)
    prompt_user = load_prompt(prompt_user)
    # prompts are loaded, dataset is loaded, now we need to judge the answers.
    questions = dataset["question"]
    oracle_answers = dataset["oracle_answer"]
    chunks = dataset["chunk"]
    summaries = dataset["summary"]
    answers_a = dataset["answer_a"]
    answers_b = dataset["answer_b"]

    # make prompts
    prompts = []
    for i in range(len(questions)):
        prompt = prompt_user.format(question=questions[i], oracle_answer=oracle_answers[i], chunk=chunks[i], summary=summaries[i], answer_a=answers_a[i], answer_b=answers_b[i])
        prompts.append(prompt)
    messages = [{"role": "system", "content": prompt_system}] + [{"role": "user", "content": prompt} for prompt in prompts]
    results = run_parallel_inference(messages, config)

    # now, we need to save the results
    # add new columns to the dataset
    dataset = dataset.add_column("judge_full_result", results)

    pass
