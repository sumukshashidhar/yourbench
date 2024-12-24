import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
from loguru import logger


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
    return config["configurations"]["hf_organization"] + "/" + config["selected_choices"]["reformat_for_judging"]["source_dataset_name"]


def reformat_for_judging(config: dict):
    """
    Reformat a large answer dataset into a a vs b format for judging purposes.
    """
    # load the dataset
    dataset = load_dataset(_get_full_dataset_name_for_questions(config), split="train")

    # get the configurations
    candidate_a = (config["selected_choices"]["reformat_for_judging"]["a"]["model"],
                  config["selected_choices"]["reformat_for_judging"]["a"]["answer_scenario"])
    candidate_b = (config["selected_choices"]["reformat_for_judging"]["b"]["model"],
                  config["selected_choices"]["reformat_for_judging"]["b"]["answer_scenario"])

    # convert to pandas
    df = dataset.to_pandas()

    # Create a list to store the reformatted rows
    reformatted_rows = []

    # group by question id
    for question_id, group in df.groupby("question_id"):
        # get the answers for both candidates
        answer_a = group[(group["generating_model"] == candidate_a[0]) &
                        (group["scenario"] == candidate_a[1])]
        answer_b = group[(group["generating_model"] == candidate_b[0]) &
                        (group["scenario"] == candidate_b[1])]

        if not answer_a.empty and not answer_b.empty:
            # Create a new row with all original columns from answer_a
            new_row = answer_a.iloc[0].to_dict()
            # Add the answers and scenarios
            new_row['answer_a'] = new_row.pop('answer')  # Rename original answer to answer_a
            new_row['answer_b'] = answer_b.iloc[0]['answer']
            new_row['answer_a_scenario'] = str(candidate_a)
            new_row['answer_b_scenario'] = str(candidate_b)
            reformatted_rows.append(new_row)

    # Create new dataframe and convert to HuggingFace dataset
    reformatted_df = pd.DataFrame(reformatted_rows)

    # Drop the specified columns
    columns_to_drop = ['scenario', 'generating_model', 'full_response']
    reformatted_df = reformatted_df.drop(columns=columns_to_drop, errors='ignore')

    reformatted_dataset = Dataset.from_pandas(reformatted_df)

    handle_dataset_push(reformatted_dataset, config["selected_choices"]["reformat_for_judging"]["target_dataset_name"], config)
