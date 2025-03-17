"""
Loading Engine Module

This module provides utility functions to load configuration files for tasks,
with support for environment variable substitution.
"""

import os
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from loguru import logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the task configuration from a YAML file, substituting environment variables.

    This function reads a YAML configuration file, expands any environment variables
    present (using the '$VAR' syntax), and returns the configuration as a dictionary.

    Parameters:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The configuration loaded as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file could not be found at config_path.
        yaml.YAMLError: If there was an error parsing the YAML content.
    """
    # Load environment variables from .env files
    load_dotenv()

    if not os.path.exists(config_path):
        logger.error("Configuration file not found: {}", config_path)
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        # Read the raw configuration file
        with open(config_path, "r") as file:
            config_str = file.read()
        logger.debug(
            "Successfully read configuration file from {}",
            config_path)

        # Substitute environment variables in the configuration string
        expanded_config_str = os.path.expandvars(config_str)

        # Parse the YAML configuration
        config = yaml.safe_load(expanded_config_str)
        logger.info("Configuration loaded successfully from {}", config_path)
        return config

    except Exception as exc:
        logger.exception("Failed to load configuration due to: {}", exc)
        raise