# 🤗 Yourbench

**Dynamic Evaluation Set Generation for LLM Benchmarking [NAACL '25*]*

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![🤗 Hugging Face](https://img.shields.io/badge/huggingface-datasets-yellow)](https://huggingface.co/docs/datasets)

</div>

## 🌟 Overview

Yourbench is a powerful framework for dynamically generating evaluation sets from source documents. It addresses the limitations of static benchmarks and benchmark saturation by creating diverse, contextually-rich questions tailored to specific educational levels.

### 🔄 Process Flow

![Process Flow](static/images/process-figure.png)

## ✨ Features

- 🔄 **Dynamic Generation**: Create evaluation sets on-the-fly from any source documents
- 📚 **Semantic Chunking**: Smart document splitting that maintains context and meaning
- 🤔 **Multi-hop Questions**: Generate questions that require synthesizing information across document sections
- 📊 **Configurable Difficulty**: Tailor questions to specific educational levels
- 🔍 **Diverse Question Types**: Support for 10 different question types
- 🤖 **Model Flexibility**: Works with OpenAI and Azure OpenAI models via LiteLLM
- 📦 **Hugging Face Integration**: Direct dataset publishing to Hugging Face Hub

## 🛠️ Requirements

- Python 3.12+
- [LiteLLM](https://github.com/BerriAI/litellm) for model inference
- [Sentence Transformers](https://www.sbert.net/) for semantic chunking
- [Hugging Face Datasets](https://huggingface.co/docs/datasets) for dataset management
- OpenAI API Compatible API / Azure AI. (more model types coming soon!)

## 📦 Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

1. Set up your environment:
```bash
# For OpenAI / OpenAI compatible APIs
export MODEL_BASE_URL=your_openai_url
export MODEL_API_KEY=your_openai_key

# For Azure OpenAI
export AZURE_BASE_URL=your_azure_url
export AZURE_API_KEY=your_azure_key
```

2. Create a task configuration (`config.yaml`). [Here is some more information!](docs/configuration.md). You can also look at an [example task configuration](task_configs/yourbench_y1/config.yaml)

3. Run the example task (after setting your 🤗 username / organization in the config!):
```bash
python yourbench/main.py --task-name yourbench_y1
```

## 📚 Documentation

Detailed documentation is available in the `docs` directory:

- [Configuration Guide](docs/configuration.md): Comprehensive guide to YAML configuration
- [Question Generation](docs/question_generation.md): Details about the question generation process
- [Chunking System](docs/chunking.md): Information about the semantic chunking system

## 🏗️ Pipeline Components

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

## 🎯 Question Types

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

## ⚙️ Configuration

Example configuration:

```yaml
task_name: yourbench_y1
configurations:
  push_to_huggingface: true
  set_hf_repo_visibility: public
  hf_organization: your-org
  model:
    model_name: gpt-4
    model_type: openai
    max_concurrent_requests: 512

selected_choices:
  generate_dataset:
    execute: true
    files_directory: examples/data
    dataset_name: my_dataset
```

See [Configuration Guide](docs/configuration.md) for detailed options.

## 🧰 Development

We use:
- [Ruff](https://github.com/astral-sh/ruff) for code formatting and linting
- [pytest](https://docs.pytest.org/) for testing


## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies
4. Make your changes
5. Run tests and ensure code style compliance
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LiteLLM](https://github.com/BerriAI/litellm) for model inference
- [Sentence Transformers](https://www.sbert.net/) for semantic embeddings
- [Hugging Face](https://huggingface.co/) for dataset infrastructure
