from datasets import load_dataset, Dataset
import pandas as pd

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

    # Group by the specified columns
    grouped = df.groupby(['title', 'chunk', 'question', 'question_type', 'estimated_difficulty', 'citations']).agg({
        'answer': lambda x: list(x),
        'generating_model': lambda x: list(x),
        'scenario': lambda x: list(x)
    }).reset_index()

    # Create new dataframe with desired format
    new_rows = []
    for _, row in grouped.iterrows():
        answers_dict = {(model, scenario): answer 
                       for model, scenario, answer in zip(row['generating_model'], 
                                                        row['scenario'], 
                                                        row['answer'])}
        
        # Find answers for both scenarios
        answer_a = answers_dict.get(candidate_a, "")
        answer_b = answers_dict.get(candidate_b, "")
        
        if answer_a and answer_b:  # Only include if we have both answers
            new_rows.append({
                'title': row['title'],
                'chunk': row['chunk'],
                'question': row['question'],
                'question_type': row['question_type'],
                'estimated_difficulty': row['estimated_difficulty'],
                'citations': row['citations'],
                'generating_model_a_scenario': str(candidate_a),  # Convert tuple to string
                'generating_model_b_scenario': str(candidate_b),  # Convert tuple to string
                'answer_a': answer_a,
                'answer_b': answer_b
            })

    # Create new DataFrame
    new_df = pd.DataFrame(new_rows)
    
    # Convert to HuggingFace dataset
    hf_dataset = Dataset.from_pandas(new_df)

    # push to hub
    hf_dataset.push_to_hub(config["configurations"]["hf_organization"] + "/" + config["selected_choices"]["reformat_for_judging"]["target_dataset_name"])




