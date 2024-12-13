# Yourbench

![Yourbench Logo](static/images/yourbench.jpg)

Yourbench is a powerful framework for dynamically generating evaluation sets from source documents. It addresses the limitations of static benchmarks and benchmark saturation by creating diverse, contextually-rich questions tailored to specific educational levels.

## Features

- 🔄 **Dynamic Generation**: Create evaluation sets on-the-fly from any source documents
- 📚 **Semantic Chunking**: Smart document splitting that maintains context and meaning
- 🤔 **Multi-hop Questions**: Generate questions that require synthesizing information across document sections
- 📊 **Configurable Difficulty**: Tailor questions to specific educational levels
- 🔍 **Diverse Question Types**: Support for 10 different question types
- 🤖 **Model Flexibility**: Works with OpenAI and Azure OpenAI models
- 📦 **Hugging Face Integration**: Direct dataset publishing to Hugging Face Hub

## Documentation

Detailed documentation is available in the `docs` directory:

- [Configuration Guide](docs/configuration.md): Comprehensive guide to YAML configuration
- [Question Generation](docs/question_generation.md): Details about the question generation process
- [Chunking System](docs/chunking.md): Information about the semantic chunking system

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

1. Create a task configuration:
```yaml
task_name: my_task
configurations:
  push_to_huggingface: true
  model:
    model_name: your-model
    model_type: openai
    max_concurrent_requests: 32
```

2. Run the task:
```bash
python src/yourbench/run_task.py --task-name my_task
```

## Pipeline Components

Yourbench consists of several modular components that can be enabled or disabled through configuration:

### 1. Dataset Generation
- Processes source documents
- Creates structured datasets
- Supports local files and Hugging Face datasets

### 2. Document Summarization
- Generates document summaries
- Provides context for question generation
- Uses configured language model

### 3. Semantic Chunking
- Splits documents intelligently
- Maintains semantic coherence
- Configurable chunk sizes and overlap

### 4. Multi-hop Chunk Creation
- Pairs related document chunks
- Enables complex reasoning questions
- Smart chunk selection

### 5. Question Generation
- Single-shot questions from individual chunks
- Multi-hop questions from chunk pairs
- 10 different question types
- Difficulty calibration
- Educational level targeting

### 6. Dataset Management
- Hugging Face integration
- Local storage options
- Dataset versioning

## Question Types

1. **Analytical**: Break down complex ideas
2. **Application-based**: Apply concepts to scenarios
3. **Clarification**: Deep dive into specifics
4. **Counterfactual**: Explore alternatives
5. **Conceptual**: Examine theories
6. **True-false**: Verify understanding
7. **Factual**: Test recall
8. **Open-ended**: Encourage discussion
9. **False-premise**: Correct misconceptions
10. **Edge-case**: Test boundaries

## Configuration

Yourbench uses YAML configuration files for task definition. Example:

```yaml
task_name: yourbench_y1
configurations:
  push_to_huggingface: true
  set_hf_repo_visibility: public
  hf_organization: your-org
  model:
    model_name: model-name
    model_type: openai
    max_concurrent_requests: 512

selected_choices:
  generate_dataset:
    execute: true
    files_directory: examples/data
    dataset_name: my_dataset
```

See [Configuration Guide](docs/configuration.md) for detailed options.

## Environment Setup

Required environment variables:

For OpenAI:
```bash
export MODEL_BASE_URL=your_openai_url
export MODEL_API_KEY=your_openai_key
```

For Azure:
```bash
export AZURE_BASE_URL=your_azure_url
export AZURE_API_KEY=your_azure_key
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

