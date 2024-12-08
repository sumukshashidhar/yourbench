from pathlib import Path
from typing import Any, Dict

import yaml


def load_task_config(task_name: str) -> Dict[str, Any]:
    """
    Loads task configuration from test_simple_dataset for a given task name.

    Args:
        task_name: Name of the task to load configuration for

    Returns:
        Dictionary containing task configuration

    Raises:
        FileNotFoundError: If task config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = (
        Path(__file__).parent.parent.parent.parent
        / "task_configs"
        / task_name
        / "config.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(f"Task config not found at: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse config file: {e}")

    return config


def get_available_tasks() -> list[str]:
    """Returns a list of all available task names in test_simple_dataset"""
    dataset_path = Path(__file__).parent.parent.parent.parent / "task_configs"
    return [
        d.name
        for d in dataset_path.iterdir()
        if d.is_dir() and (d / "config.yaml").exists()
    ]
