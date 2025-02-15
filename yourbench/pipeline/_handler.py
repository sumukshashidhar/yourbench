"""Pipeline execution handler module.

This module is responsible for loading and executing pipeline stages based on
configuration settings. It dynamically imports and runs pipeline stages while
providing proper error handling and logging.
"""

from typing import Dict, Any
from yourbench.utils.loading_engine import load_config
from loguru import logger
import importlib

def run_pipeline(config_file_path: str) -> None:
    """Execute the pipeline stages defined in the configuration file.

    Args:
        config_file_path: Path to the configuration file containing pipeline settings.

    Raises:
        FileNotFoundError: If the config file cannot be found.
        Exception: For any other unexpected errors during pipeline execution.
    """
    try:
        # Load pipeline configuration
        logger.debug(f"Loading configuration from {config_file_path}")
        config: Dict[str, Any] = load_config(config_file_path)
        pipeline_configuration = config.get("pipeline", {})

        if not pipeline_configuration:
            logger.warning("No pipeline stages configured")
            return

        # Execute pipeline stages
        for stage_name, stage_configuration in pipeline_configuration.items():
            try:
                if not stage_configuration.get("run", True):
                    logger.info(f"Skipping stage: {stage_name}")
                    continue

                logger.info(f"Starting execution of stage: {stage_name}")
                stage_module = importlib.import_module(f"yourbench.pipeline.{stage_name}")
                stage_module.run(config)
                logger.success(f"Successfully completed stage: {stage_name}")
            except Exception as e:
                logger.error(f"Error executing pipeline stage {stage_name}: {str(e)}")
                raise

    except Exception as e:
        logger.critical(f"Pipeline execution failed: {str(e)}")
        raise