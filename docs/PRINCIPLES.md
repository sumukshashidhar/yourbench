## Process Flow

![YourBench pipeline process flow diagram – from document ingestion to evaluation](assets/yourbench_pipeline.png)

Under the hood, YourBench follows a multi-stage pipeline to turn raw documents into a ready-to-use benchmark dataset:

1. **Document Ingestion** – Convert PDFs, HTML, Word docs, or raw text files into a standardized format (Markdown) for downstream processing.
2. **Summarization** – Generate a concise *global summary* of each document using a designated summarization model. This helps distill key points and limit the scope for question generation.
3. **Chunking** – Split documents into smaller chunks (and optionally merge small pieces) based on semantic similarity or length constraints. This ensures long or complex documents are broken into manageable sections for Q\&A generation.
4. **Question Generation** – For each chunk (or combination of chunks), generate questions:

   * *Single-Hop:* Create straightforward questions answerable from a single chunk.
   * *Multi-Hop:* Combine multiple chunks to produce more complex questions that require integrating information from different parts of the content.
5. **Deduplication** – Remove or group together near-duplicate questions using embedding-based similarity, to avoid redundant entries in your benchmark.
6. **Analysis** – Evaluate the question set for coverage and difficulty. YourBench provides logging and analysis tools to measure how well the questions cover the source content, the distribution of topics, estimated difficulty levels, etc., and can run custom analysis modules.
7. **Export** – Finally, output the generated Q\&A benchmark. The results can be saved as a local dataset (using the Hugging Face `datasets` format) or even uploaded to the Hugging Face Hub for sharing. This makes it easy to evaluate models on the new benchmark or even set up a public leaderboard.

Throughout this process, **YourBench ensures the questions are grounded in your provided documents**, rather than what an LLM might already know. By using documents (and even an optional fresh document dataset like *Tempora-0325* for time-sensitive topics), the pipeline minimizes reliance on a model’s parametric memory, yielding more truthful and up-to-date evaluation queries.

Want to understand **how to configure the pipeline?** Check out the [Configuration Guide](./CONFIGURATION.md).

Want to know more about the **columns in the output dataset?** Check out the [Dataset Columns](./DATASET_COLUMNS_DESCRIPTION.md).


## Highlights


* **Dynamic Benchmark Generation** – Produce diverse, up-to-date question-answer pairs derived from real-world source documents (PDF, Word, HTML, even multimedia).
* **Scalable & Structured** – Seamlessly handle ingestion, summarization, and multi-hop chunking for large or specialized datasets.
* **Extensible Pipeline** – Use out-of-the-box stages (ingestion, summarization, question generation) or plug in custom models and logic to accommodate domain-specific needs.
* **Robust Configuration** – Control the entire pipeline via a single YAML config (model choices, data paths, chunking parameters, generation prompts, deduplication thresholds, etc.).
* **Multi-Model Support** – Assign different LLMs for each stage (ingestion, summarization, QG, answering), fostering broader coverage and question-style diversity.
* **Deduplication & Quality Filtering** – Automatically group near-duplicates to prune questions and retain a curated set of high-quality queries.
* **Logging & Analysis** – Built-in metrics evaluate dataset coverage, question distribution, difficulty, and more.
* **Flexible Output** – Save generated benchmarks locally or push them to the Hugging Face Hub for sharing or public leaderboards.

<div align="center">

<a href="https://youtu.be/mhszO6kZSbI" target="_blank">
  <img src="https://img.youtube.com/vi/mhszO6kZSbI/maxresdefault.jpg" alt="YourBench Demo Video" width="600" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" />
  <br/>
  <img src="https://img.shields.io/badge/Watch%20Demo-YouTube-red?style=for-the-badge&logo=youtube" alt="Watch Demo on YouTube"/>
  <br/>
  <em>Watch our 3-minute demo of the YourBench pipeline</em>
</a>
</div>