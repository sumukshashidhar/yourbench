"""Utility for loading custom Pydantic schemas from file paths.

Schema files must export a Pydantic BaseModel named `DataFormat`.
Users specify the path: `path/to/schema.py`
"""

import importlib.util
from typing import Type
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from yourbench.utils.question_schemas import get_default_schema


# Standard class name that all schema files must export
SCHEMA_CLASS_NAME = "DataFormat"


class SchemaLoadError(ValueError):
    """Raised when schema loading fails."""

    pass


def load_schema_from_spec(spec: str | None, question_mode: str) -> Type[BaseModel]:
    """Load a Pydantic schema from a file path or return the default.

    Args:
        spec: Path to schema file (must export `DataFormat` class), or None.
        question_mode: The question mode ('open-ended' or 'multi-choice').

    Returns:
        A Pydantic BaseModel subclass.

    Raises:
        SchemaLoadError: If the spec is invalid or the class cannot be loaded.
    """
    if not spec:
        return get_default_schema(question_mode)

    file_path = Path(spec).resolve()
    if not file_path.exists():
        raise SchemaLoadError(f"Schema file not found: {file_path}")

    if not file_path.suffix == ".py":
        raise SchemaLoadError(f"Schema file must be a Python file (.py): {file_path}")

    # Load the module dynamically
    module_name = file_path.stem
    spec_loader = importlib.util.spec_from_file_location(module_name, file_path)
    if spec_loader is None or spec_loader.loader is None:
        raise SchemaLoadError(f"Cannot load module from: {file_path}")

    module = importlib.util.module_from_spec(spec_loader)
    spec_loader.loader.exec_module(module)

    # Get the DataFormat class from the module
    if not hasattr(module, SCHEMA_CLASS_NAME):
        raise SchemaLoadError(
            f"Class '{SCHEMA_CLASS_NAME}' not found in {file_path}. "
            f"Available: {[n for n in dir(module) if not n.startswith('_')]}"
        )

    schema_cls = getattr(module, SCHEMA_CLASS_NAME)

    # Validate it's a Pydantic model
    if not isinstance(schema_cls, type) or not issubclass(schema_cls, BaseModel):
        raise SchemaLoadError(
            f"'{SCHEMA_CLASS_NAME}' in {file_path} must be a Pydantic BaseModel subclass, got {type(schema_cls)}"
        )

    logger.info(f"Loaded custom schema: {SCHEMA_CLASS_NAME} from {file_path}")
    return schema_cls
