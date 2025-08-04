# YourBench Prompts

This directory contains all the prompt templates used throughout the YourBench pipeline. Prompts are organized by their function and stored as markdown files for easy maintenance and version control.

## Directory Structure

- `ingestion/` - Prompts for document ingestion and processing
  - `pdf_llm_prompt.md` - Prompt for PDF content extraction with LLM assistance

- `summarization/` - Prompts for document summarization
  - `summarization_user_prompt.md` - Main summarization prompt
  - `combine_summaries_user_prompt.md` - Prompt for combining multiple summaries

- `question_generation/` - Prompts for generating questions
  - `single_shot_system_prompt.md` - System prompt for single-hop open-ended questions
  - `single_shot_system_prompt_multi.md` - System prompt for single-hop multiple-choice questions
  - `single_shot_user_prompt.md` - User prompt for single-hop questions
  - `multi_hop_system_prompt.md` - System prompt for multi-hop questions
  - `multi_hop_user_prompt.md` - User prompt for multi-hop questions

- `question_rewriting/` - Prompts for improving question quality
  - `question_rewriting_system_prompt.md` - System prompt for question rewriting
  - `question_rewriting_user_prompt.md` - User prompt for question rewriting

- `qa/` - Prompts for question answering and evaluation
  - `zeroshot_qa_user_prompt.md` - Zero-shot QA prompt
  - `gold_qa_user_prompt.md` - QA prompt with context
  - `judge_answer_system_prompt.md` - System prompt for answer evaluation
  - `judge_answer_user_prompt.md` - User prompt for answer evaluation

## Usage

These prompts are loaded automatically by the YourBench configuration system. You can override any prompt by specifying a custom file path or inline content in your configuration YAML file.

## Customization

To use custom prompts, you can either:
1. Provide a file path to your custom prompt in the configuration
2. Provide the prompt content directly in the configuration
3. Place your custom prompt files in the appropriate directory (for development)