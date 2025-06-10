import os
import csv
import atexit
import datetime
import collections
from typing import Dict, List

import tiktoken
from loguru import logger


# Using defaultdict for easier accumulation
_cost_data = collections.defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0})
_individual_log_file = os.path.join("logs", "inference_cost_log_individual.csv")
_aggregate_log_file = os.path.join("logs", "inference_cost_log_aggregate.csv")
_individual_header_written = False


def _get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Gets a tiktoken encoding, defaulting to cl100k_base with fallback."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to get encoding '{encoding_name}'. Falling back to 'cl100k_base'. Error: {e}")
        return tiktoken.get_encoding("cl100k_base")


def _ensure_logs_dir():
    """Ensures the logs directory exists."""
    os.makedirs("logs", exist_ok=True)


def _count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    """Counts tokens in a single string."""
    if not text:
        return 0
    try:
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return 0


def _count_message_tokens(messages: List[Dict[str, str]], encoding: tiktoken.Encoding) -> int:
    """Counts tokens in a list of messages, approximating OpenAI's format."""
    num_tokens = 0
    # Approximation based on OpenAI's cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    # This might not be perfectly accurate for all models/providers but is a reasonable estimate.
    tokens_per_message = 3
    tokens_per_name = 1

    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if value:
                num_tokens += _count_tokens(str(value), encoding)
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def _log_individual_call(model_name: str, input_tokens: int, output_tokens: int, tags: List[str], encoding_name: str):
    """Logs a single inference call's cost details."""
    global _individual_header_written
    try:
        _ensure_logs_dir()
        is_new_file = not os.path.exists(_individual_log_file)
        mode = "a" if not is_new_file else "w"

        with open(_individual_log_file, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header only if the file is new or header wasn't written yet in this run
            if is_new_file or not _individual_header_written:
                writer.writerow(["timestamp", "model_name", "stage", "input_tokens", "output_tokens", "encoding_used"])
                _individual_header_written = True

            stage = ";".join(tags) if tags else "unknown"
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            writer.writerow([timestamp, model_name, stage, input_tokens, output_tokens, encoding_name])
    except Exception as e:
        logger.error(f"Failed to write to individual cost log: {e}")


def _update_aggregate_cost(model_name: str, input_tokens: int, output_tokens: int):
    """Updates the global dictionary for aggregate costs."""
    try:
        _cost_data[model_name]["input_tokens"] += input_tokens
        _cost_data[model_name]["output_tokens"] += output_tokens
        _cost_data[model_name]["calls"] += 1
    except Exception as e:
        logger.error(f"Failed to update aggregate cost data: {e}")


def _write_aggregate_log():
    """Writes the aggregated cost data to a file at program exit."""
    try:
        if not _cost_data:
            logger.info("No cost data collected, skipping aggregate log.")
            return

        _ensure_logs_dir()
        logger.info(f"Writing aggregate cost log to {_aggregate_log_file}")
        with open(_aggregate_log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model_name", "total_input_tokens", "total_output_tokens", "total_calls"])
            for model_name, data in sorted(_cost_data.items()):
                writer.writerow([model_name, data["input_tokens"], data["output_tokens"], data["calls"]])
        logger.success(f"Aggregate cost log successfully written to {_aggregate_log_file}")
    except Exception as e:
        # Use print here as logger might be shutting down during atexit
        print(f"ERROR: Failed to write aggregate cost log: {e}", flush=True)


# Register the aggregate log function to run at exit
atexit.register(_write_aggregate_log)
