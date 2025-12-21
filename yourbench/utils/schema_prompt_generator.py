"""Generate prompt instructions from Pydantic schemas.

This module converts Pydantic model definitions into clear, structured
instructions for LLMs to produce valid JSON output.
"""

import json
from typing import Any, Type, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


def _extract_constraints(field_info: FieldInfo) -> dict[str, Any]:
    """Extract constraint values from field metadata."""
    constraints = {}
    for item in field_info.metadata:
        cls_name = type(item).__name__
        if cls_name == "Ge":
            constraints["ge"] = item.ge
        elif cls_name == "Le":
            constraints["le"] = item.le
        elif cls_name == "MinLen":
            constraints["min_length"] = item.min_length
        elif cls_name == "MaxLen":
            constraints["max_length"] = item.max_length
    return constraints


def _get_type_description(annotation: Any) -> str:
    """Convert a Python type annotation to a human-readable description."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Literal types first
    if origin is Literal:
        values = get_args(annotation)
        if len(values) <= 5:
            return "one of: " + ", ".join(f'"{v}"' for v in values)
        return "one of: " + ", ".join(f'"{v}"' for v in values[:5]) + ", ..."

    # Handle common generic types
    if origin is list:
        inner = _get_type_description(args[0]) if args else "any"
        return f"array of {inner}"

    if origin is dict:
        return "object"

    # Handle basic types
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"

    # Fallback
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _format_field(name: str, field_info: FieldInfo, annotation: Any) -> str:
    """Format a single field as a description line."""
    type_desc = _get_type_description(annotation)
    description = field_info.description or "No description provided."

    # Extract constraints from metadata
    meta = _extract_constraints(field_info)
    constraint_strs = []
    if "ge" in meta:
        constraint_strs.append(f"min: {meta['ge']}")
    if "le" in meta:
        constraint_strs.append(f"max: {meta['le']}")
    if "min_length" in meta:
        constraint_strs.append(f"min items: {meta['min_length']}")
    if "max_length" in meta:
        constraint_strs.append(f"max items: {meta['max_length']}")

    constraint_str = f" ({', '.join(constraint_strs)})" if constraint_strs else ""

    return f"- `{name}` ({type_desc}{constraint_str}): {description}"


def generate_schema_instructions(schema: Type[BaseModel]) -> str:
    """Generate prompt instructions from a Pydantic model."""
    lines = []

    # Add class docstring if present
    if schema.__doc__:
        lines.append(schema.__doc__.strip())
        lines.append("")

    lines.append("Output a JSON array where each object has the following fields:")
    lines.append("")

    # Process each field
    for field_name, field_info in schema.model_fields.items():
        annotation = field_info.annotation
        lines.append(_format_field(field_name, field_info, annotation))

    lines.append("")
    lines.append("Wrap your JSON output in `<output_json>` tags.")

    return "\n".join(lines)


def generate_example_json(schema: Type[BaseModel]) -> str:
    """Generate an example JSON structure from a Pydantic model."""
    example = {}

    for field_name, field_info in schema.model_fields.items():
        annotation = field_info.annotation
        origin = get_origin(annotation)

        if annotation is str:
            example[field_name] = "<string>"
        elif annotation is int:
            meta = _extract_constraints(field_info)
            ge_val = meta.get("ge")
            le_val = meta.get("le")
            if ge_val is not None and le_val is not None:
                example[field_name] = (ge_val + le_val) // 2
            else:
                example[field_name] = 0
        elif annotation is float:
            example[field_name] = 0.0
        elif annotation is bool:
            example[field_name] = True
        elif origin is list:
            example[field_name] = ["<item>"]
        elif origin is dict:
            example[field_name] = {}
        elif origin is Literal:
            values = get_args(annotation)
            example[field_name] = values[0] if values else "<value>"
        else:
            example[field_name] = "<value>"

    return json.dumps([example], indent=2)
