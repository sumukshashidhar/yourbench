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

Yourbench can be easily configured by defining YAML configuration files. You can find examples of these in the `task_configs` directory. Define a folder for your task, and add a `config.yaml` file. You can then run the task by calling `python src/yourbench/run_task.py --task-name <task_name>`.

To test the dataset generation pipeline, you can run the following command, after specifying your own huggingface username or organization (`hf_organization`) in the `config.yaml` file. 

Example:

```bash
python src/yourbench/run_task.py --task-name yourbench_y1
```


### Pipeline Parts

YourBench is highly modular, and each part of the pipeline can be swapped out for a different implementation. This allows for easy experimentation with different approaches to generation. You can also run individual parts of the pipeline independently, based on your needs:

Below is a list of the current pipeline parts, in order of expected execution:

- **Dataset Generation**: Given some source documents, you can generate a new huggingface dataset in the required format.
- **Summarization**: Given a structured huggingface dataset, you can generate summaries with your model of choice.
- **Chunking**: Given a structured huggingface dataset, you can generate semantically meaningful chunks of text, with configurable parameters.
- **Multihop Chunking**: Given a chunked dataset, you can combine multiple chunks from the same document, or different documents, randomly to create multihop questions.
- **Single Shot Question Generation**: Given a chunked dataset, you can generate single shot questions based on a taxonomy and test audience, with your model of choice. 
- **Multihop Question Generation**: Given a chunked, multihop dataset, you can generate multihop questions with your model of choice.
- **Question Rephrasing**: Given a question dataset, you can rephrase questions to introduce more variety and challenge downstream evaluations.
- **Question Reweighting**: Given a large question dataset, you can reweight the questions to reduce the overall size while maintaining the diversity of the dataset.
- **Question Answering**: Given a question and a chunked dataset, you can answer the question with the chunked dataset.
- **Evaluation**: Given a question  and a chunked dataset, you can evaluate the question answering performance of a model.

### YAML Configuration

#### Top-Level Structure

The YAML file is structured as a dictionary with the following top-level keys:

* `task_name`: The name of the task, which will be used to identify the task in the logs and output files.
* `configurations`: 
* `datasets`: A list of datasets to be used in the task. These are either huggingface datasets, or local directories containing markdown files.
* `models`: The model to be used in the task.

