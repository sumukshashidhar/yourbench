# YourBench

![YourBench Logo](static/images/yourbench.jpg)

YourBench is a framework to dynamically generate reliable evaluation sets from source documents.

## Usage

### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Configuration

Yourbench can be easily configured by defining YAML configuration files. You can find examples of these in the `task_configs` directory. Define a folder for your task, and add a `config.yaml` file. You can then run the task by calling `python src/yourbench/run_task.py --task-name <path_to_task_config>`.

Example:

```bash
python src/yourbench/run_task.py --task-name test_simple_dataset
```

### Pipeline Parts

YourBench is highly modular, and each part of the pipeline can be swapped out for a different implementation. This allows for easy experimentation with different approaches to generation. You can also run individual parts of the pipeline independently, based on your needs:

Below is a list of the current pipeline parts:

- **Dataset Generation**: Given some source documents, you can generate a new huggingface dataset in the required format.
- **Summarization**: Given a structured, huggingface dataset in the required foramt, you can generate summaries with your model of choice.
- **Chunking**: Given a structured, huggingface dataset in the required foramt, you can generate semantically meaningful chunks of text, with configurable parameters.
- **Multihop Chunking**: Given a chunked dataset, you can combine multiple chunks from the same document, or different documents, randomly to create multihop questions.
- **Single Shot Question Generation**: Given a chunked dataset, you can generate single shot questions based on a taxonomy and test audience, with your model of choice. 
- **Multihop Question Generation**: Given a chunked, multihop dataset, you can generate multihop questions with your model of choice.
- **Question Rephrasing**: Given a question dataset, you can rephrase questions to introduce more variety and challenge downstream evaluations.
- **Question Answering**: Given a question and a chunked dataset, you can answer the question with the chunked dataset.
- **Evaluation**: Given a question  and a chunked dataset, you can evaluate the question answering performance of a model.