import random

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
    Reformat a large answer dataset into an a-vs-b format for judging purposes,
    then randomly swap A and B for exactly 50% of the rows.
    """
    # 1) Load the dataset
    dataset = load_dataset(_get_full_dataset_name_for_questions(config), split="train")

    # 2) Get the configurations
    candidate_a = (config["selected_choices"]["reformat_for_judging"]["a"]["model"],
                   config["selected_choices"]["reformat_for_judging"]["a"]["answer_scenario"])
    candidate_b = (config["selected_choices"]["reformat_for_judging"]["b"]["model"],
                   config["selected_choices"]["reformat_for_judging"]["b"]["answer_scenario"])

    # 3) Convert to pandas
    df = dataset.to_pandas()

    # 4) Build the new, reformatted rows
    reformatted_rows = []
    for question_id, group in df.groupby("question_id"):
        # Get the answers for both candidates
        answer_a = group[
            (group["generating_model"] == candidate_a[0]) &
            (group["scenario"] == candidate_a[1])
        ]
        answer_b = group[
            (group["generating_model"] == candidate_b[0]) &
            (group["scenario"] == candidate_b[1])
        ]

        if not answer_a.empty and not answer_b.empty:
            # Create a new row with all columns from answer_a (just pick the first row)
            new_row = answer_a.iloc[0].to_dict()

            # Rename original "answer" to "answer_a"
            new_row["answer_a"] = new_row.pop("answer")
            # Add the B answer
            new_row["answer_b"] = answer_b.iloc[0]["answer"]

            # Add scenario strings
            new_row["answer_a_scenario"] = str(candidate_a)
            new_row["answer_b_scenario"] = str(candidate_b)

            reformatted_rows.append(new_row)

    reformatted_df = pd.DataFrame(reformatted_rows)

    # 5) Drop unused columns
    columns_to_drop = ["scenario", "generating_model", "full_response"]
    reformatted_df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    # 6) Randomly swap A/B in exactly 50% of the rows
    #    (if there's an odd number of rows, this will floor() it)
    n_rows = len(reformatted_df)
    half = n_rows // 2  # integer division

    # Shuffle row indices
    all_indices = list(range(n_rows))
    random.shuffle(all_indices)

    # Pick half the indices to invert
    invert_indices = all_indices[:half]

    # Swap function
    def swap_answers(df, idx):
        temp_answer = df.at[idx, "answer_a"]
        temp_scenario = df.at[idx, "answer_a_scenario"]

        df.at[idx, "answer_a"] = df.at[idx, "answer_b"]
        df.at[idx, "answer_a_scenario"] = df.at[idx, "answer_b_scenario"]

        df.at[idx, "answer_b"] = temp_answer
        df.at[idx, "answer_b_scenario"] = temp_scenario

    for idx in invert_indices:
        swap_answers(reformatted_df, idx)

    # 7) Convert back to Hugging Face Dataset
    reformatted_dataset = Dataset.from_pandas(reformatted_df)

    # 8) Push or save locally
    handle_dataset_push(
        reformatted_dataset,
        config["selected_choices"]["reformat_for_judging"]["target_dataset_name"],
        config
    )

    # --- Small Test / Logging ---
    # Let’s log the percentage that was inverted.
    inverted_percentage = (len(invert_indices) / n_rows) * 100 if n_rows > 0 else 0
    logger.info(f"Randomly inverted {inverted_percentage:.2f}% of the dataset ("
                f"{len(invert_indices)}/{n_rows} rows).")
