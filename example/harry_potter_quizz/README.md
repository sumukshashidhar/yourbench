# Getting started with YourBench

_Generating a dataset with YourBench - So magic that you'll make Harry Potter jealous and Hermione proud_

### What we're going to build

We will craft a multiple-choice questions dataset about Harry Potter stories.
To do this, we will use [YourBench](https://github.com/huggingface/yourbench), a library from Hugging Face. It can generate high-quality benchmark datasets by ingesting various source documents. Starting from an existing corpus of text makes it useful to reduce hallucinations, as the LLMs will ground synthetic questions and answers on the text. YourBench makes this process so simple that you'll make Harry Potter jealous.

### Prerequisites

1. We recommend Python >= 3.12
2. Create a virtual environment and [install yourbench](https://github.com/huggingface/yourbench?tab=readme-ov-file#installation)
3. Download a PDF file containing a summary of each Harry Potter book collected from Wikipedia (you can find it [here](https://raw.githubusercontent.com/patrickfleith/test-files/main/Harry_Potter_Wikipedia_Plots.pdf)) and save it in the `data` directory. Or you can get it with the following `wget` command in your working directory:
  ```bash
  mkdir -p data && wget https://raw.githubusercontent.com/patrickfleith/test-files/main/Harry_Potter_Wikipedia_Plots.pdf -O data/Harry_Potter_Wikipedia_Plots.pdf
  ``` 
4. Create a `.env` file in our working directory. We'll need a Hugging Face token with write access to our private datasets if we want to push it to the Hub, and the API keys of the model provider. In our case, we'll use OpenRouter as it's very convenient for utilizing any proprietary or open-source model:
  ```bash
  HF_TOKEN=hf_xxxxxx
  OPENROUTER_API_KEY=sk-xxxxxxx
  ```
  
- Not sure how to get an Hugging Face token? [Check this](https://huggingface.co/docs/hub/en/security-tokens)
- Also need an OPENROUTER_API_KEY? [It's here](https://openrouter.ai/settings/keys)

### How does it work?

#### Configure, then run

Using YourBench is as simple as:

1. Writing a YAML configuration for the generation pipeline (see below)
2. Running a command line: `yourbench run config.yaml`

#### Under the hood

We'll configure YourBench to follow a multi-stage pipeline to turn raw documents into a ready-to-use benchmark dataset:

1. **Document Ingestion** – Convert our PDFs into a standardized format (Markdown) for downstream processing (note that YourBench can handle many other formats like HTML, Word docs, or raw text files)
2. **Summarization** – Generate a concise *global summary* of each document using a designated summarization model. This helps distill key points and limit the scope for question generation.
3. **Chunking** – Split documents into smaller chunks (and optionally merge small pieces) based on length constraints. This ensures long or complex documents are broken into manageable sections for Q\&A generation.
4. **Question Generation** – For each chunk (or combination of chunks), we'll generate multiple-choice questions. In our case, we'll only generate single-hop questions (meaning questions answerable given a single chunk), but YourBench offers more options like multi-hop question generation and even cross-document question generation. This is beyond our needs here.
5. **Export** – Finally, output the generated Q\&A benchmark. The results can be saved as a local dataset (using the Hugging Face `datasets` format) or even uploaded to the Hugging Face Hub for sharing. This makes it easy to evaluate models on the new benchmark or even set up a public leaderboard.

Throughout this process, **YourBench ensures the questions are grounded in our provided documents**, rather than what an LLM might already know. It also allows you to create fresh questions based on new documents.

### Let's configure our pipeline

#### 1. Configuration file structure

We create a `config.yaml` file in our working directory.

A typical YourBench configuration file follows this structure:

```yaml
hf_configuration:
  # Hugging Face dataset settings

model_list:
  # List of model configurations

model_roles:
  # Optional: Assign specific models to pipeline stages

pipeline:
  # Pipeline stage configurations
```

#### 2. Hugging Face Configuration

We need to configure the Hugging Face dataset settings. We'll use the `harry-potter-quizz` dataset name and set it to private. We'll also use the environment variables for the Hugging Face organization and token.

The `hf_organization` field is optional. If we don't specify it, YourBench will resolve it based on the token and can also push the dataset under our username.

```yaml
hf_configuration:
  hf_dataset_name: harry-potter-quizz
  private: true
  hf_organization: $HF_ORGANIZATION
  hf_token: $HF_TOKEN
```

#### 3. Model Configuration

We'll use OpenRouter to access the `gpt-oss-120b` model. The model configuration specifies which LLM to use for the various pipeline stages. In our case, we keep it simple: the same model is used for all the pipeline stages.

```yaml
model_list:
  - model_name: openai/gpt-oss-120b
    base_url: https://openrouter.ai/api/v1
    api_key: $OPENROUTER_API_KEY
    max_concurrent_requests: 8
```

#### 4. Pipeline Configuration

Now we configure each stage of the pipeline.

> [!NOTE]
> We are only generating questions from a single chunk here using `single_shot_question_generation` stage. There are two modes in YourBench: `multi-choice` and `open-ended`.
> - `multi-choice` mode: Generate multiple-choice questions from each chunk (includes the correct choice and the incorrect choices).
> - `open-ended` mode: Generate open-ended questions from each chunk (and the expected correct answer).

```yaml
pipeline:
  ingestion:
    source_documents_dir: data
    output_dir: processed
  
  summarization:
    # Uses default settings and the model from model_list
  
  chunking:
    l_max_tokens: 1024        # Maximum number of tokens per chunk
    token_overlap: 256        # Token overlap between chunks
  
  single_shot_question_generation:
    question_mode: multi-choice
```


### Making Harry Potter jealous! ✨

*Now we have everything to run YourBench*

```bash
yourbench run config.yaml
```

This command will:

1. **Ingest** the Harry Potter PDF and convert it to a processed markdown format
2. **Summarize** the content to extract key themes
3. **Chunk** the text into optimal sizes for question generation
4. **Generate** multiple-choice questions grounded in the source material
5. **Upload** the final dataset to Hugging Face Hub 

*→ We just made a dataset of multiple-choice questions from our document in less than 1 minute, enough to make Harry Potter jealous*

But wait, there's more...

### Making Hermione Proud

*TL;DR: We need to look at the data!*

Once the pipeline is run, we can look at our data through the Hugging Face Hub (the dataset viewer can take several minutes to load properly).

We will notice that our dataset is made of several subsets. Each corresponds to one of the pipeline stages we ran:
- ingested
- summarized
- chunked
- single_shot_questions

To load the dataset that we just uploaded to the hub, we use `load_dataset` from the `datasets` library, and specify which subset we want to load with the parameter `name`:

```python
from datasets import load_dataset
harry_potter_quizz = load_dataset("your_hf_organization/harry-potter-quizz", name='single_shot_questions')
```

#### Understand the `single_shot_questions` subset

The `single_shot_questions` subset contains the following relevant columns for our multiple-choice quiz use case:

- `question`: The question generated, for instance: *What is Professor Snape's true intention during Harry's first Quidditch match?*
- `choices`: A list of choices generated, for instance: ['(A) He is trying to protect Harry by jinxing his broom.', '(B) He wants to sabotage Harry so Gryffindor will lose.', '(C) He is unaware of the match and does nothing.', "(D) He is testing Harry's flying skills."] 
- `answer`: The letter corresponding to the correct answer, for instance: "A"
- `chunk_id`: The id of the chunk used to generate the question. We'll be able to find the chunk in the `chunked` subset.

**Congratulations!** We've successfully created a grounded, high-quality benchmark dataset from raw documents. The magic of YourBench has transformed the Harry Potter Wikipedia Plots PDF into a comprehensive quiz dataset that would make even Hermione proud! 🧙‍♀️✨

<details>
<summary><strong>Want to know more about the columns in the other dataset subsets?</strong></summary>

#### Understand the `ingested` subset

The ingested dataset contains the following columns:
- `document_id`: A unique identifier for each file we ingested
- `document_filename`: The name of the file
- `document_text`: The whole text content parsed from the file
- `document_metadata`: Metadata about the file such as the file size.

If we ingested from 1 document, there is just one row.

#### Understand the `summarized` subset

The summarized dataset contains the same columns as the `ingested` dataset, plus the following:
- `document_summary`: An LLM-generated summary of the document
- `summarization_model`: The model used to generate the summary

If we ingested from 1 document, there is just one row.

#### Understand the `chunked` subset

The ingested document is split into chunks of 1024 tokens with an overlap of 256 tokens. 
The number of rows is still equal to the number of ingested documents. This is because each row contains again all of the above mentioned (document_id, document_filename, document_text, document_metadata, document_summary, summarization_model). 

**All the chunks for that document are stored in that row in the `chunks` column**. In addition each row["chunks"] is a dictionary with:
- `chunk_id`: The ID of the chunk. This ID reuses the document_id and increments a suffix to make it unique. The first chunk has chunk_id of document_id_0, the second chunk has chunk_id of document_id_1, and so on.
- `chunk_text`: The actual text content of the chunk, which is used to generate synthetic questions and answer pairs.

**Multi-hop chunks**
- `multihop_chunks`: these are combinations of chunks for multi-hop question generation pipelines (although not used in this example). This is a dictionary with the following keys:
    - `chunk_ids`: A list of chunk IDs
    - `chunks_text`: A list of the chunk texts

</details>
