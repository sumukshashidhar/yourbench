"""Clean prompt template substitution for question generation.

This module handles the substitution of placeholders in prompt templates
with schema-specific instructions, examples, and reminders.
"""

from typing import Type

from pydantic import BaseModel

from yourbench.utils.schema_prompt_generator import (
    generate_example_json,
    generate_critical_reminders,
    generate_schema_instructions,
)


def build_system_prompt(template: str, schema: Type[BaseModel]) -> str:
    """Substitute placeholders in a prompt template with schema-specific content.

    Replaces:
    - {schema_definition} with the schema field definitions
    - {example_output} with example JSON output
    - {critical_reminders} with field-specific reminders

    If no placeholders are found, returns the template unchanged.
    """
    # Generate schema-specific content
    schema_def = f"""## Output Format

Output a JSON array wrapped in `<output_json>` tags.

{generate_schema_instructions(schema)}"""

    example_json = generate_example_json(schema)
    example_output = f"""## Example Output

<document_analysis>
[Your analysis of the document content here]
</document_analysis>

<output_json>
{example_json}
</output_json>"""

    critical_reminders = generate_critical_reminders(schema)

    # Perform substitutions
    result = template.replace("{schema_definition}", schema_def)
    result = result.replace("{example_output}", example_output)
    result = result.replace("{critical_reminders}", critical_reminders)

    return result
