## Process Flow

![YourBench pipeline process flow diagram – from document ingestion to evaluation](docs/assets/yourbench_pipeline.png)

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