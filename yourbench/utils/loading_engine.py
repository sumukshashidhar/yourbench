"""Utilities for loading YAML configuration files with environment expansion."""

import os
from typing import Any
from pathlib import Path
from dataclasses import dataclass

import yaml
from dotenv import load_dotenv
from loguru import logger


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables in nested structures."""
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(i) for i in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Convenience wrapper returning ``ConfigLoader(config_path).load()``."""
    return ConfigLoader(Path(config_path)).load()


@dataclass(slots=True)
class ConfigLoader:
    path: Path

    def load(self) -> dict[str, Any]:
        load_dotenv()

        if not self.path.is_file():
            logger.error(f"Configuration file not found: {self.path}")
            raise FileNotFoundError(self.path)

        try:
            text = self.path.read_text()
            logger.debug(f"Read configuration from {self.path}")
            expanded = os.path.expandvars(text)
            config = yaml.safe_load(expanded) or {}
            result = _expand_env_vars(config)
            logger.debug(f"Configuration loaded successfully from {self.path}")
            return result
        except yaml.YAMLError as exc:
            logger.error(f"Error parsing YAML {self.path}: {exc}")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to load configuration {self.path}: {exc}")
            raise
