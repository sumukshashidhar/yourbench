"""Utility for loading custom Pydantic schemas from file paths.

Users can specify a path like `path/to/schema.py:ClassName` in their config
to use a custom question schema.
"""

import importlib.util
from typing import Type
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from yourbench.utils.question_schemas import get_default_schema


class SchemaLoadError(ValueError):
    """Raised when schema loading fails."""

    pass


def load_schema_from_spec(spec: str | None, question_mode: str) -> Type[BaseModel]:
    """Load a Pydantic schema from a spec string or return the default.

    Args:
        spec: Schema specification in format `path/to/file.py:ClassName` or None.
        question_mode: The question mode ('open-ended' or 'multi-choice').

    Returns:
        A Pydantic BaseModel subclass.

    Raises:
        SchemaLoadError: If the spec is invalid or the class cannot be loaded.
    """
    if not spec:
        return get_default_schema(question_mode)

    if ":" not in spec:
        raise SchemaLoadError(f"Invalid schema spec: '{spec}'. Expected format: 'path/to/file.py:ClassName'")

    file_path_str, class_name = spec.rsplit(":", 1)
    file_path = Path(file_path_str).resolve()

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

    # Get the class from the module
    if not hasattr(module, class_name):
        raise SchemaLoadError(
            f"Class '{class_name}' not found in {file_path}. "
            f"Available: {[n for n in dir(module) if not n.startswith('_')]}"
        )

    schema_cls = getattr(module, class_name)

    # Validate it's a Pydantic model
    if not isinstance(schema_cls, type) or not issubclass(schema_cls, BaseModel):
        raise SchemaLoadError(
            f"'{class_name}' in {file_path} must be a Pydantic BaseModel subclass, got {type(schema_cls)}"
        )

    logger.info(f"Loaded custom schema: {class_name} from {file_path}")
    return schema_cls
