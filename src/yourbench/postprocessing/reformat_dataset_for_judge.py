import pandas as pd
from datasets import Dataset, load_dataset


def _get_full_dataset_name_for_questions(config: dict) -> str:
    return config["configurations"]["hf_organization"] + "/" + config["selected_choices"]["reformat_for_judging"]["source_dataset_name"]


def reformat_for_judging(config: dict):
    """
    Reformat a large answer dataset into a a vs b format for judging purposes.
    """
    # load the dataset
    dataset = load_dataset(_get_full_dataset_name_for_questions(config), split="train")

    # get the configurations
    candidate_a = (config["selected_choices"]["reformat_for_judging"]["a"]["model"], config["selected_choices"]["reformat_for_judging"]["a"]["answer_scenario"])
    candidate_b = (config["selected_choices"]["reformat_for_judging"]["b"]["model"], config["selected_choices"]["reformat_for_judging"]["b"]["answer_scenario"])

    # convert to pandas
    df = dataset.to_pandas()

    # Define columns to aggregate
    agg_columns = ['answer', 'generating_model', 'scenario']

    # Group by all columns except the ones we aggregate
    group_columns = [col for col in df.columns if col not in agg_columns]

    # Group by the remaining columns
    grouped = df.groupby(group_columns).agg({
        'answer': lambda x: list(x),
        'generating_model': lambda x: list(x),
        'scenario': lambda x: list(x)
    }).reset_index()

    # Add debug prints
    print(f"Original dataset size: {len(df)}")
    print(f"Looking for answers from candidates: {candidate_a} and {candidate_b}")

    # Group by question and model, ignoring scenario
    candidate_a_model = candidate_a[0]  # Just use the model name
    candidate_b_model = candidate_b[0]

    new_rows = []
    for _, row in grouped.iterrows():
        # Create a dictionary that only keys by model
        answers_by_model = {}
        for model, scenario, answer in zip(row['generating_model'],
                                         row['scenario'],
                                         row['answer']):
            answers_by_model[model] = (answer, scenario)

        # Find answers for both models, regardless of scenario
        if candidate_a_model in answers_by_model and candidate_b_model in answers_by_model:
            answer_a, scenario_a = answers_by_model[candidate_a_model]
            answer_b, scenario_b = answers_by_model[candidate_b_model]

            new_rows.append({
                'title': row['title'],
                'summary': row['summary'],
                'chunk': row['chunk'],
                'question': row['question'],
                'oracle_answer': row['oracle_answer'],
                'question_type': row['question_type'],
                'estimated_difficulty': row['estimated_difficulty'],
                'citations': row['citations'],
                'generating_model_a_scenario': f"({candidate_a_model}, {scenario_a})",
                'generating_model_b_scenario': f"({candidate_b_model}, {scenario_b})",
                'answer_a': answer_a,
                'answer_b': answer_b
            })

    print(f"Final dataset size: {len(new_rows)}")

    # Create new DataFrame
    new_df = pd.DataFrame(new_rows)

    # Convert to HuggingFace dataset
    hf_dataset = Dataset.from_pandas(new_df)

    # push to hub
    hf_dataset.push_to_hub(config["configurations"]["hf_organization"] + "/" + config["selected_choices"]["reformat_for_judging"]["target_dataset_name"])
