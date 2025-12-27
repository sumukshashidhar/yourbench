# Rich PDF Extraction with Gemini

This example is to demonstrate how you can perform LLM ingestion of a PDF, on a per page basis. This preserves rich charts, figures, diagrams, latex, etc, while doing the question generation process.

In this example, we use `gemini-2.5-flash` through [OpenRouter](https://openrouter.ai/) to process and generate questions.

## How to run?

```bash
# set an OPENROUTER_API_KEY
export OPENROUTER_API_KEY=
# run the script
yourbench run example/rich_pdf_extraction_with_gemini/config.yaml
```

The expected result from this run can be found at the following huggingface dataset: [yourbench/mckinsey_state_of_ai_doc_understanding](https://huggingface.co/datasets/yourbench/mckinsey_state_of_ai_doc_understanding)