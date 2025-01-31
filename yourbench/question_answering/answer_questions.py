from datasets import Dataset, load_dataset
from utils.dataset_engine import handle_dataset_push, make_dataset_name
from utils.inference_engine import run_parallel_inference
from utils.load_prompt import load_prompt
from utils.parsing_engine import extract_content_from_xml_tags


def _generate_scenario_answers(
    config: dict,
    *,
    # Where to load the dataset from:
    load_dataset_func,
    # How to figure out final "answer" dataset name:
    resolve_dataset_name_func,
    # Which prompt path/key to load:
    prompt_path,
    # A function to build each prompt string from a row index:
    prompt_builder,
    # Scenario label to store in `scenario` column:
    scenario_label,
    # A function to build the dictionary of row data for the new dataset:
    row_builder,
):
    """
    Consolidates the common logic of:
      1) load dataset
      2) create prompts
      3) run inference
      4) parse answers
      5) build final dataset
      6) push that dataset
    """
    # 1) load dataset
    dataset = load_dataset_func()
    # temp, only keep 2 rows
    dataset = dataset.select(range(2))
    if isinstance(dataset, dict):
        # If load_dataset returned a dict with split keys, use `train`
        dataset = dataset["train"]

    # 2) create the base prompt
    prompt_template = load_prompt(prompt_path)

    # 3) build the user messages (prompts) for run_parallel_inference
    messages = []
    for i in range(len(dataset)):
        # scenario-specific logic for building a single prompt string
        prompt_str = prompt_builder(dataset, i, prompt_template)
        messages.append([{"role": "user", "content": prompt_str}])

    # 4) run inference in parallel
    responses = run_parallel_inference(messages, config)

    # 5) parse answers from the responses
    answers = [extract_content_from_xml_tags(r, "answer") for r in responses]

    # 6) build dataset rows
    dataset_rows = []
    for i in range(len(dataset)):
        row = row_builder(dataset, i, responses[i], answers[i], scenario_label, config)
        dataset_rows.append(row)

    # 7) form the new huggingface Dataset
    new_dataset = Dataset.from_list(dataset_rows)

    # 8) figure out final dataset name
    final_dataset_name = resolve_dataset_name_func(config)

    # 9) push the dataset
    handle_dataset_push(config, final_dataset_name, new_dataset)


#
# BELOW: Each scenario is a thin wrapper around _generate_scenario_answers
#


def _get_zeroshot_answers(config: dict) -> None:
    """
    The scenario that currently 'works as expected.'
    We'll replicate its logic in a DRY-friendly manner.
    """

    def load_zeroshot_dataset():
        # Matches original zero_shot logic
        ds_name = make_dataset_name(
            config,
            config["pipeline"]["answer_questions_with_llm"]["source_dataset_name"],
        )
        return load_dataset(ds_name)["train"]

    def resolve_zeroshot_name(cfg):
        return make_dataset_name(
            cfg,
            cfg["pipeline"]["answer_questions_with_llm"]["answer_scenarios"]["zero_shot"]["answer_dataset_name"],
        )

    def build_prompt(dataset, i, prompt_template):
        question = dataset["question"][i]
        return prompt_template.format(question=question)

    def build_row(dataset, i, full_response, answer, scenario_label, cfg):
        # Exactly as in the original _get_zeroshot_answers
        return {
            "question_id": dataset["question_id"][i],
            "document_id": dataset["document_id"][i],
            "document_name": dataset["document_name"][i],
            "document_category": dataset["document_category"][i],
            "document_summary": dataset["document_summary"][i],
            "chunk_location_id": dataset["chunk_location_id"][i],
            "chunk": dataset["chunk"][i],
            "test_audience": dataset["test_audience"][i],
            "full_generation_response": full_response,
            "document_analysis": dataset["document_analysis"][i],
            "question_type": dataset["question_type"][i],
            "thought_process": dataset["thought_process"][i],
            "question": dataset["question"][i],
            "oracle_answer": dataset["answer"][i],  # from original code
            "answer": answer,
            "scenario": scenario_label,
            "generating_model": cfg["configurations"]["model"]["model_name"],
        }

    _generate_scenario_answers(
        config,
        load_dataset_func=load_zeroshot_dataset,
        resolve_dataset_name_func=resolve_zeroshot_name,
        prompt_path=f"{config['pipeline']['answer_questions_with_llm']['prompt_prefix']}.fast_answer_q_zeroshot_user",
        prompt_builder=build_prompt,
        scenario_label="zero_shot",
        row_builder=build_row,
    )


def _get_zeroshot_cot_answers(config: dict) -> None:
    """
    Zero-shot with chain-of-thought scenario.
    """

    def load_dataset_func():
        # Follows original pattern (we assume same dataset as the typical "full" approach)
        ds_name = make_dataset_name(
            config,
            config["pipeline"]["answer_questions_with_llm"]["source_dataset_name"],
        )
        return load_dataset(ds_name)["train"]

    def resolve_dataset_name(cfg):
        # from the code: scenario is "zero_shot_with_cot"
        return cfg["pipeline"]["answer_questions_with_llm"]["answer_scenarios"]["zero_shot_with_cot"][
            "answer_dataset_name"
        ]

    def build_prompt(dataset, i, prompt_template):
        question = dataset["question"][i]
        chunk = dataset["chunk"][i]
        return prompt_template.format(question=question, document=chunk)

    def build_row(dataset, i, full_response, answer, scenario_label, cfg):
        # match the original `_get_zeroshot_cot_answers` structure
        return {
            "question_id": dataset["question_id"][i],
            "document_id": dataset["document_id"][i],
            "document_name": dataset["document_name"][i],
            "document_category": dataset["document_category"][i],
            "document_summary": dataset["document_summary"][i],
            "chunk_location_id": dataset["chunk_location_id"][i],
            "chunk": dataset["chunk"][i],
            "test_audience": dataset["test_audience"][i],
            "full_generation_response": full_response,
            "document_analysis": dataset["document_analysis"][i],
            "question_type": dataset["question_type"][i],
            "thought_process": dataset["thought_process"][i],
            "question": dataset["question"][i],
            "oracle_answer": dataset["answer"][i],  # from original code
            "answer": answer,
            "scenario": scenario_label,
            "generating_model": cfg["configurations"]["model"]["model_name"],
        }

    _generate_scenario_answers(
        config,
        load_dataset_func=load_dataset_func,
        resolve_dataset_name_func=resolve_dataset_name,
        prompt_path=f"{config['pipeline']['answer_questions_with_llm']['prompt_prefix']}.fast_answer_q_cot_user",
        prompt_builder=build_prompt,
        scenario_label="zero_shot_with_cot",
        row_builder=build_row,
    )


def _get_document_summary_answers(config: dict) -> None:
    """
    Scenario: 'answer_with_document_summary'
    """

    def load_dataset_func():
        ds_name = make_dataset_name(
            config,
            config["pipeline"]["answer_questions_with_llm"]["source_dataset_name"],
        )
        return load_dataset(ds_name)["train"]

    def resolve_dataset_name(cfg):
        return cfg["pipeline"]["answer_questions_with_llm"]["answer_scenarios"]["answer_with_document_summary"][
            "answer_dataset_name"
        ]

    def build_prompt(dataset, i, prompt_template):
        question = dataset["question"][i]
        summary = dataset["document_summary"][i]
        return prompt_template.format(question=question, summary=summary)

    def build_row(dataset, i, full_response, answer, scenario_label, cfg):
        # match original `_get_document_summary_answers` row structure
        return {
            "question_id": dataset["question_id"][i],
            "document_id": dataset["document_id"][i],
            "document_name": dataset["document_name"][i],
            "document_category": dataset["document_category"][i],
            "document_summary": dataset["document_summary"][i],
            "chunk_location_id": dataset["chunk_location_id"][i],
            "chunk": dataset["chunk"][i],
            "test_audience": dataset["test_audience"][i],
            "full_generation_response": full_response,
            "document_analysis": dataset["document_analysis"][i],
            "question_type": dataset["question_type"][i],
            "thought_process": dataset["thought_process"][i],
            "question": dataset["question"][i],
            "oracle_answer": dataset["answer"][i],  # from original code
            "answer": answer,
            "scenario": scenario_label,
            "generating_model": cfg["configurations"]["model"]["model_name"],
        }

    _generate_scenario_answers(
        config,
        load_dataset_func=load_dataset_func,
        resolve_dataset_name_func=resolve_dataset_name,
        prompt_path=f"{config['pipeline']['answer_questions_with_llm']['prompt_prefix']}.fast_answer_q_docsummary_user",
        prompt_builder=build_prompt,
        scenario_label="answer_with_document_summary",
        row_builder=build_row,
    )


def _get_relevant_chunk_answers(config: dict) -> None:
    """
    Scenario: 'answer_with_relevant_chunks'
    """

    def load_dataset_func():
        ds_name = make_dataset_name(
            config,
            config["pipeline"]["answer_questions_with_llm"]["source_dataset_name"],
        )
        return load_dataset(ds_name)["train"]

    def resolve_dataset_name(cfg):
        return cfg["pipeline"]["answer_questions_with_llm"]["answer_scenarios"]["answer_with_relevant_chunks"][
            "answer_dataset_name"
        ]

    def build_prompt(dataset, i, prompt_template):
        question = dataset["question"][i]
        chunk = dataset["chunk"][i]
        return prompt_template.format(question=question, document=chunk)

    def build_row(dataset, i, full_response, answer, scenario_label, cfg):
        # match original `_get_relevant_chunk_answers` row structure
        return {
            "question_id": dataset["question_id"][i],
            "document_id": dataset["document_id"][i],
            "document_name": dataset["document_name"][i],
            "document_category": dataset["document_category"][i],
            "document_summary": dataset["document_summary"][i],
            "chunk_location_id": dataset["chunk_location_id"][i],
            "chunk": dataset["chunk"][i],
            "test_audience": dataset["test_audience"][i],
            "full_generation_response": full_response,
            "document_analysis": dataset["document_analysis"][i],
            "question_type": dataset["question_type"][i],
            "thought_process": dataset["thought_process"][i],
            "question": dataset["question"][i],
            "oracle_answer": dataset["answer"][i],  # from original code
            "answer": answer,
            "scenario": scenario_label,
            "generating_model": cfg["configurations"]["model"]["model_name"],
        }

    _generate_scenario_answers(
        config,
        load_dataset_func=load_dataset_func,
        resolve_dataset_name_func=resolve_dataset_name,
        prompt_path=f"{config['pipeline']['answer_questions_with_llm']['prompt_prefix']}.fast_answer_q_relevant_chunk_user",
        prompt_builder=build_prompt,
        scenario_label="answer_with_relevant_chunks",
        row_builder=build_row,
    )


def _get_gold_answers(config: dict) -> None:
    """
    Scenario: 'gold_standard'
    """

    def load_dataset_func():
        ds_name = make_dataset_name(
            config,
            config["pipeline"]["answer_questions_with_llm"]["source_dataset_name"],
        )
        return load_dataset(ds_name)["train"]

    def resolve_dataset_name(cfg):
        return cfg["pipeline"]["answer_questions_with_llm"]["answer_scenarios"]["gold_standard"]["answer_dataset_name"]

    def build_prompt(dataset, i, prompt_template):
        question = dataset["question"][i]
        chunk = dataset["chunk"][i]
        summary = dataset["document_summary"][i]
        return prompt_template.format(question=question, document=chunk, summary=summary)

    def build_row(dataset, i, full_response, answer, scenario_label, cfg):
        # match original `_get_gold_answers` row structure
        return {
            "question_id": dataset["question_id"][i],
            "document_id": dataset["document_id"][i],
            "document_name": dataset["document_name"][i],
            "document_category": dataset["document_category"][i],
            "document_summary": dataset["document_summary"][i],
            "chunk_location_id": dataset["chunk_location_id"][i],
            "chunk": dataset["chunk"][i],
            "test_audience": dataset["test_audience"][i],
            "full_generation_response": full_response,
            "document_analysis": dataset["document_analysis"][i],
            "question_type": dataset["question_type"][i],
            "thought_process": dataset["thought_process"][i],
            "question": dataset["question"][i],
            "oracle_answer": dataset["answer"][i],  # from original code
            "answer": answer,
            "scenario": scenario_label,
            "generating_model": cfg["configurations"]["model"]["model_name"],
        }

    _generate_scenario_answers(
        config,
        load_dataset_func=load_dataset_func,
        resolve_dataset_name_func=resolve_dataset_name,
        prompt_path=f"{config['pipeline']['answer_questions_with_llm']['prompt_prefix']}.fast_answer_q_gold_user",
        prompt_builder=build_prompt,
        scenario_label="gold_standard",
        row_builder=build_row,
    )


def answer_questions_with_llm(config: dict):
    """
    The main orchestrator. We do not optimize or reorganize it;
    we only call the scenario subroutines. Each subroutine is now
    much thinner, deferring the repeated logic to _generate_scenario_answers.
    """
    answer_scenarios = config["pipeline"]["answer_questions_with_llm"]["answer_scenarios"]

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
