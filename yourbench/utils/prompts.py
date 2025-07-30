"""
This module contains the prompts for the pipeline stages.
"""

SUMMARIZATION_USER_PROMPT = """You are an AI assistant tasked with analyzing and summarizing documents from various domains. Your goal is to generate a concise yet comprehensive summary of the given document. Follow these steps carefully:

1. You will be provided with a document extracted from a website. This document may be very long and/or split into multiple contiguous sections. It may contain unnecessary artifacts such as links, HTML tags, or other web-related elements.

2. Here is the document to be summarized:
<document>
{document}
</document>

3. Before generating the summary, use a mental scratchpad to take notes as you read through the document. Enclose your notes within <scratchpad> tags. For example:

<scratchpad>
- Main topic: [Note the main subject of the document]
- Key points: [List important information across the entire document]
- Structure: [Note how the document is organized or chunked]
- Potential artifacts to ignore: [List any web-related elements that should be disregarded]
</scratchpad>

4. As you analyze the document:
   - Focus solely on the content, ignoring any unnecessary web-related elements.
   - Treat all sections or chunks as part of a single, continuous document.
   - Identify the main topic and key points from the entire input.
   - Pay attention to the overall structure and flow of the document.

5. After your analysis, generate a final summary that:
   - Captures the essence of the document in a concise manner.
   - Includes the main topic and key points.
   - Presents information in a logical and coherent order.
   - Is comprehensive yet concise, typically ranging from 3-5 sentences (unless the document is particularly long or complex).

6. Enclose your final summary within <final_summary> tags. For example:

<final_summary>
[Your concise and comprehensive summary of the document goes here.]
</final_summary>

Remember, your task is to provide a clear, accurate, and concise summary of the document's content, disregarding any web-related artifacts or unnecessary elements. For long documents, ensure your summary reflects the complete scope and structure of the content."""


QUESTION_GENERATION_SYSTEM_PROMPT_HEADER = """## Your Role

You are an expert educational content designer who crafts thoughtful, research-oriented **question–answer pairs** from supplied text. Your questions must be moderately challenging, promote reflection and nuanced understanding, and respect any constraints in the input.

---

## Input Structure

The input **always** contains these tags in this exact order (do **not** rename, remove, or reorder them):

```
<additional_instructions>
…
</additional_instructions>

<title>
…
</title>

<document_summary>
…
</document_summary>

<text_chunk>
…
</text_chunk>
```

---

## Primary Objective

From the single `<text_chunk>`, create a set of self-contained, research-level question–answer pairs that:

* Encourage deep engagement and critical thought.
* Demonstrate clear pedagogical value.
* Align with any directives in `<additional_instructions>`.
* Sit at a **moderate difficulty** (≈ 4-7 on a 1-10 scale).

---

## Workflow

Enclose all private reasoning in one pair of `<document_analysis>` tags, then output the finished question–answer pairs **outside** those tags.

Inside `<document_analysis>`:

1. **Comprehension** – Identify the key ideas, arguments, methods, and findings in `<text_chunk>`.
2. **Depth Search** – Note implicit assumptions, subtle details, and potential applications.
3. **Complexity Calibration** – Select an overall difficulty rating (1-10) that matches the learning goals.
4. **Question Planning** – Map each question to a specific learning objective or insight.
5. **Irrelevance Filter** – Ignore hyperlinks, ads, navigation text, disclaimers, or nonsensical passages. If the entire `<text_chunk>` is irrelevant, explain why and **do not** produce questions.

---

## Question Guidelines

* **Tone** – Natural, engaging, and conversational.
* **Clarity** – Each question and answer must be understandable without external references.
* **Types** – Choose whichever of the following best fits the content (you need not use them all): analytical, application-based, conceptual, clarification, counterfactual, edge-case, true/false, factual, open-ended, false-premise.
* **Context** – Provide enough information in the question for it to stand alone, yet avoid unnecessary repetition.

---

## Handling Irrelevant or Bogus Content

* Explicitly ignore non-informational elements (ads, footers, social-media buttons, etc.).
* If only portions are irrelevant, use the meaningful parts and note exclusions in `<document_analysis>`.
* If the entire `<text_chunk>` lacks educational value, document that decision in `<document_analysis>` and output **no** questions.

---

**Do not change the input or output format.** All internal reasoning stays within `<document_analysis>`; the learner sees only the polished question–answer pairs that follow it.
"""

QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT = """## Output Structure

This prompt is used exclusively for generating **open-ended** questions.

Present your final output as a list of JSON objects strictly adhering to this Pydantic model, wrapped within `<output_json>` XML tags:

```python
class QuestionRow(BaseModel):
    thought_process: str # Clear, detailed rationale for selecting question and analysis approach
    question_type: Literal["analytical", "application-based", "clarification",
                           "counterfactual", "conceptual", "true-false",
                           "factual", "open-ended", "false-premise", "edge-case"]
    question: str  # The generated question
    answer: str  # Full answer to the question
    estimated_difficulty: int  # Difficulty level from 1 (easy) to 10 (very difficult), calibrated according to additional instructions
    citations: List[str]  # Direct quotes from the text_chunk supporting the answer
```

## Output Format

Begin by thoughtfully analyzing the provided text_chunk within <document_analysis> XML tags.
Then present the resulting list of QuestionRow objects in proper JSON format inside <output_json> XML tags.

## Example:

<document_analysis>
Key concept: Semantic chunking and its effect on information retrieval
Facts: Semantic chunking groups semantically similar sentences within token limits
Reasoning cues: Relevance of chunk boundaries for downstream QA tasks
</document_analysis>

<output_json>
[
  {
    "thought_process": "The question evaluates whether the model understands how semantic chunking contributes to retrieval quality. It encourages reflection on how coherence impacts model outputs.",
    "question_type": "open-ended",
    "question": "How does semantic chunking improve information retrieval performance in large document processing?",
    "answer": "Semantic chunking improves retrieval by preserving contextual coherence, allowing models to access more relevant and interpretable chunks during downstream tasks like question answering.",
    "estimated_difficulty": 6,
    "citations": [
      "Semantic chunking groups related sentences within token boundaries.",
      "Coherent chunks help downstream tasks focus on relevant context."
    ],
  },
  ...
]
</output_json>
"""

QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT_MULTI = """## Output Structure

Present your final output as JSON objects strictly adhering to this schema, enclosed within `<output_json>` XML tags. This structure supports both open-ended and multiple-choice questions.

```python
class QuestionRow(BaseModel):
   thought_process: str  # Explanation for why this question was generated, including reasoning or distractor logic
   question_type: Literal["analytical", "application-based", "clarification",
                           "counterfactual", "conceptual", "true-false",
                           "factual", "false-premise", "edge-case"]
   question: str  # The question text
   answer: str  # One of "A", "B", "C", or "D"
   choices: List[str]  # Must contain exactly 4 items
   estimated_difficulty: int  # Integer between 1 (easy) and 10 (difficult)
   citations: List[str]  # Supporting quotes or phrases from the text
```

## Output Format

Start with a thoughtful analysis of the <text_chunk> wrapped inside <document_analysis> tags. Identify key concepts, reasoning paths, and challenging content.

Then output a list of well-structured questions in valid JSON syntax inside <output_json> tags.

## Example:

<document_analysis>
Key concept: Semantic chunking and its role in preprocessing
Facts: Chunking maintains coherence based on token and semantic similarity
Reasoning cues: Trade-offs in chunk size and overlap
</document_analysis>

<output_json>
[
  {
    "thought_process": "This question targets a conceptual understanding of why semantic chunking is needed. Distractors reflect common misconceptions.",
    "question_type": "conceptual",
    "question": "What is the primary benefit of using semantic chunking in document processing?",
    "answer": "B",
    "choices": [
      "(A) It compresses documents by removing white space.",
      "(B) It groups related content within token constraints for coherence.",
      "(C) It translates the document into a semantic graph.",
      "(D) It removes all non-ASCII characters for parsing."
    ],
    "estimated_difficulty": 6,
    "citations": ["Semantic chunking partitions documents into coherent segments based on semantic similarity and token length constraints."]
  },
  ...
]
</output_json>"""

QUESTION_GENERATION_SYSTEM_PROMPT_FOOTER = """## Important Notes
- Strive to generate questions that inspire genuine curiosity, reflection, and thoughtful engagement.
- Maintain clear, direct, and accurate citations drawn verbatim from the provided text_chunk.
- Ensure complexity and depth reflect thoughtful moderation as guided by the additional instructions.
- Each "thought_process" should reflect careful consideration and reasoning behind your question selection.
- Ensure rigorous adherence to JSON formatting and the provided Pydantic validation model.
- When generating questions, NEVER include phrases like 'as per the text,' 'according to the document,' or any similar explicit references. Questions should inherently integrate content naturally and stand independently without explicit references to the source material
"""

QUESTION_GENERATION_SYSTEM_PROMPT = (
    QUESTION_GENERATION_SYSTEM_PROMPT_HEADER
    + QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT
    + QUESTION_GENERATION_SYSTEM_PROMPT_FOOTER
)
QUESTION_GENERATION_SYSTEM_PROMPT_MULTI = (
    QUESTION_GENERATION_SYSTEM_PROMPT_HEADER
    + QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT_MULTI
    + QUESTION_GENERATION_SYSTEM_PROMPT_FOOTER
)

QUESTION_GENERATION_USER_PROMPT = """<title>
{title}
</title>

<document_summary>
{document_summary}
</document_summary>

<text_chunk>
{text_chunk}
</text_chunk>

<additional_instructions>
{additional_instructions}
</additional_instructions>"""


MULTI_HOP_QUESTION_GENERATION_SYSTEM_HEADER = """## Your Role

You are an expert educational content designer who crafts insightful, research-level **multi-hop question–answer pairs** from supplied text. Each question must require integrative reasoning across multiple chunks, promote moderate challenge, and respect any constraints in the input.

---

## Input Structure

The input **always** contains these tags in this exact order (do **not** rename, remove, or reorder them):

```
<additional_instructions>
…
</additional_instructions>

<title>
…
</title>

<document_summary>
…
</document_summary>

<text_chunks>
  <text_chunk_0>
  …
  </text_chunk_0>
  <text_chunk_1>
  …
  </text_chunk_1>
  [More <text_chunk_n> as needed]
</text_chunks>
```

---

## Primary Objective

From the set of `<text_chunks>`, create self-contained, multi-hop question–answer pairs that:

* Demand synthesis of information from **at least two** different chunks.
* Encourage deep engagement, critical thought, and nuanced understanding.
* Align with directives in `<additional_instructions>`.
* Sit at a **moderate difficulty** (≈ 4-7 on a 1-10 scale).

---

## Workflow

Enclose all private reasoning in one pair of `<document_analysis>` tags, then output the finished question–answer pairs **outside** those tags.

Inside `<document_analysis>`:

1. **Cross-Chunk Comprehension** – Identify key ideas, arguments, and data in each chunk.
2. **Connection Mapping** – Trace how concepts, evidence, or implications in different chunks intersect.
3. **Complexity Calibration** – Select an overall difficulty rating (1-10) that meets learning goals.
4. **Question Planning** – For each planned question, specify the chunks it links and the insight it targets.
5. **Irrelevance Filter** – Ignore ads, headers, footers, navigation text, or nonsensical passages. If a chunk is wholly irrelevant, document that and exclude it from questioning.

If **all** chunks lack educational value, explain why and **do not** generate questions.

---

## Question Guidelines

* **Multi-Hop Integration** – Each question must clearly require information from multiple chunks.
* **Tone** – Natural, engaging, and conversational.
* **Clarity** – Questions and answers must be understandable without external references.
* **Types** – Choose whichever of these best fit (no need to use all): analytical, application-based, conceptual, clarification, counterfactual, edge-case, true/false, factual, open-ended, false-premise.
* **Context** – Include enough detail for standalone sense, but avoid unnecessary repetition.

---

## Handling Irrelevant or Bogus Content

* **Exclude** navigation links, ads, promotional blurbs, or other non-informational text.
* If a chunk is partly irrelevant, use only its meaningful parts and note exclusions in `<document_analysis>`.
* If a chunk is entirely irrelevant, record that decision and skip it.
* Never force questions from unsuitable content; prioritize quality and pedagogical value.

---

**Do not change the input or output format.** All internal reasoning stays within `<document_analysis>`; learners see only the polished question–answer pairs that follow it."""


MULTI_HOP_QUESTION_GENERATION_SYSTEM_FOOTER = """## Important Notes
- Prioritize depth and thoughtfulness in your reasoning paths.
- Allow natural complexity to guide question formulation, aiming for moderate challenge.
- Precisely cite verbatim excerpts from text chunks.
- Clearly communicate your thought process for integrative reasoning.
- Adhere strictly to JSON formatting and Pydantic validation requirements.
- Generate questions that genuinely inspire deeper reflection or meaningful exploration of the provided content.
- When generating questions, NEVER include phrases like 'as per the text,' 'according to the document,' or any similar explicit references. Questions should inherently integrate content naturally and stand independently without explicit references to the source material"""

MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT = (
    MULTI_HOP_QUESTION_GENERATION_SYSTEM_HEADER
    + QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT
    + MULTI_HOP_QUESTION_GENERATION_SYSTEM_FOOTER
)
MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT_MULTI = (
    MULTI_HOP_QUESTION_GENERATION_SYSTEM_HEADER
    + QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT_MULTI
    + MULTI_HOP_QUESTION_GENERATION_SYSTEM_FOOTER
)

MULTI_HOP_QUESTION_GENERATION_USER_PROMPT = """<title>
{title}
</title>

<document_summary>
{document_summary}
</document_summary>

<text_chunks>
{chunks}
</text_chunks>

<additional_instructions>
{additional_instructions}
</additional_instructions>"""


ZEROSHOT_QA_USER_PROMPT = """Answer the following question:

<question>
{question}
</question>

Enclose your full answer in <answer> XML tags. For example:

<answer>
[your answer here]
</answer>"""

GOLD_QA_USER_PROMPT = """Answer the following question:

<question>
{question}
</question>

Here is a summary of the document the question is asked from which may be helpful:

<document_summary>
{summary}
</document_summary>

And here is a relevant chunk of the document which may prove useful

<document>
{document}
</document>

Enclose your full answer in <answer> XML tags. For example:

<answer>
[your answer here]
</answer>"""

JUDGE_ANSWER_SYSTEM_PROMPT = """You will be provided with the summary of a document, a piece of text, a question generated from that text, and the correct or "gold" answer to the question. Additionally, you will receive two answers: Answer A and Answer B. Your task is to determine which of these answers is closer to the gold answer by assessing the overlap of key points between the ground truth and the two given answers.

# Steps

1. **Document Understanding**:
   - Analyze the provided document summary to grasp the context and main themes.

2. **Chunk Understanding**:
   - Examine the provided text (chunk) to understand its content.

3. **Question Understanding**:
   - Interpret the given question to fully comprehend what is being asked.

4. **Ground Truth Answer Understanding**:
   - Understand the provided ground truth answer, identifying its key points.

5. **Answer A Understanding**:
   - Analyze Answer A, identifying key points and assessing accuracy and factuality.

6. **Answer B Understanding**:
   - Examine Answer B, identifying key points and assessing accuracy and factuality.

7. **Similarity Comparison**:
   - Compare Answer A and the ground truth answer, noting similarities in key points.
   - Compare Answer B and the ground truth answer, noting similarities in key points.

8. **Final Similarity Analysis**:
   - Evaluate both answers based on the similarities identified and determine which is closer to the ground truth in terms of key points and factuality.

# Output Format

- Provide your final evaluation of which answer is closer to the ground truth within `<final_answer>` XML tags.
- Include a detailed analysis for each part within the designated XML tags: `<document_understanding>`, `<chunk_understanding>`, `<question_understanding>`, `<ground_truth_answer_understanding>`, `<answer_a_understanding>`, `<answer_b_understanding>`, `<similarity_comparison_answer_a>`, `<similarity_comparison_answer_b>`, and `<final_similarity_analysis>`.

# Examples

**Input**:
```xml
<document_summary>
[Summary]
</document_summary>

<piece_of_text>
[Text]
</piece_of_text>

<question>
[Question]
</question>

<gold_answer>
[Gold Answer]
</gold_answer>

<answer_a>
[Answer A]
</answer_a>

<answer_b>
[Answer B]
</answer_b>
```
**Output**:
```xml

<document_understanding>
Understanding of the summary including key themes
</document_understanding>

<chunk_understanding>
Analysis of the piece of text
</chunk_understanding>

<question_understanding>
Comprehension of the question being asked
</question_understanding>

<ground_truth_answer_understanding>
Key points from the gold answer
</ground_truth_answer_understanding>

<answer_a_understanding>
Key points and accuracy of Answer A
</answer_a_understanding>

<answer_b_understanding>
Key points and accuracy of Answer B
</answer_b_understanding>

<similarity_comparison_answer_a>
Comparison notes between Answer A and the gold answer
</similarity_comparison_answer_a>

<similarity_comparison_answer_b>
Comparison notes between Answer B and the gold answer
</similarity_comparison_answer_b>

<final_similarity_analysis>
Overall analysis determining the closer answer
</final_similarity_analysis>

<final_answer>
Answer X (where X is the option you pick)
</final_answer>
```

# Notes

- Always focus on key points and factual correctness as per the ground truth.
- Avoid any biases and rely solely on the evidence presented.
- Enclose all evaluations and analyses in the specified XML tags for clarity and structure."""

JUDGE_ANSWER_USER_PROMPT = """<document_summary>
{summary}
</document_summary>

<piece_of_text>
{chunk}
</piece_of_text>

<question>
{question}
</question>

<gold_answer>
{oracle_answer}
</gold_answer>

<answer_a>
{answer_a}
</answer_a>

<answer_b>
{answer_b}
</answer_b>"""

COMBINE_SUMMARIES_USER_PROMPT = """\
You will receive a list of chunk-level summaries from the *same* \
document.  Combine them into a single, well-structured paragraph that reads \
naturally and eliminates redundancy.

<chunk_summaries>
{chunk_summaries}
</chunk_summaries>

Return ONLY the final text inside <final_summary> tags."""

QUESTION_REWRITING_SYSTEM_PROMPT = """You are an expert at question_rewriting questions to improve their clarity, naturalness, and engagement while preserving their exact meaning and answerability.

## Your Task

Given an original question along with its answer, source text chunks, and document summary, rewrite the question following these principles:

1. **Preserve Meaning Completely**: The rewritten question must ask for exactly the same information as the original.
2. **Maintain Answerability**: The rewritten question must be answerable using the same source information.
3. **Improve Clarity**: Make the question clearer and more natural-sounding.
4. **Vary Phrasing**: Use different words and sentence structures while keeping the core query intact.
5. **Keep Appropriate Complexity**: Maintain the same level of difficulty as the original question.

## Guidelines

- DO NOT change what the question is asking for
- DO NOT add new requirements or constraints not in the original
- DO NOT remove important context or specifications from the original
- DO NOT change from open-ended to multiple-choice or vice versa
- DO make the language more conversational and engaging
- DO fix any grammatical issues in the original
- DO use synonyms and alternative phrasings
- DO maintain the same question type (factual, analytical, conceptual, etc.)

## Output Format

Provide your rewritten question within <rewritten_question> tags and a brief explanation of your question_rewriting approach within <question_rewriting_rationale> tags.

Example:
<question_rewriting_rationale>
Changed passive voice to active voice and replaced technical jargon with clearer terms while maintaining the specific focus on causal relationships.
</question_rewriting_rationale>

<rewritten_question>
[Your rewritten question here]
</rewritten_question>"""

QUESTION_REWRITING_USER_PROMPT = """Please rewrite the following question while preserving its exact meaning and answerability.

<original_question>
{original_question}
</original_question>

<answer>
{answer}
</answer>

<source_chunks>
{chunk_text}
</source_chunks>

<document_summary>
{document_summary}
</document_summary>

<additional_instructions>
{additional_instructions}
</additional_instructions>

Remember to:
1. Keep the exact same meaning and information requirements
2. Ensure the rewritten question can be answered with the same source material
3. Make the question sound more natural and engaging
4. Provide your rewritten question in <rewritten_question> tags
5. Explain your question_rewriting approach in <question_rewriting_rationale> tags"""
