from pathlib import Path
from typing import Any, Dict

import yaml


# TODO: either remove or decide how to use it
def _get_full_dataset_name_for_questions(config: dict, actual_dataset_name: str) -> str:
    return config["configurations"]["hf_organization"] + "/" + actual_dataset_name


def get_project_root() -> Path:
    """Get the project root directory (where pyproject.toml is located)"""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml found)")


# TODO: implement pydantic validation
def _validate_task_config(config: dict) -> dict:
    # check parts of the pipeline are present, else add a false for them
    return config


def load_task_config(task_name: str) -> Dict[str, Any]:
    """
    Loads task configuration for a given task name.
    """
    config_path = get_project_root() / "task_configs" / task_name / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Task config not found at: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse config file: {e}")

    return _validate_task_config(config)


def get_available_tasks() -> list[str]:
    """Returns a list of all available task names"""
    dataset_path = get_project_root() / "task_configs"
    return [d.name for d in dataset_path.iterdir() if d.is_dir() and (d / "config.yaml").exists()]
