"""
This module contains the prompts for the pipeline stages.
"""

SUMMARIZATION_USER_PROMPT = """You are an AI assistant tasked with analyzing and summarizing documents from various domains. Your goal is to generate a concise yet comprehensive summary of the given document. Follow these steps carefully:

1. You will be provided with a document extracted from a website. This document may contain unnecessary artifacts such as links, HTML tags, or other web-related elements.

2. Here is the document to be summarized:
<document>
{document}
</document>

3. Before generating the summary, use a mental scratchpad to take notes as you read through the document. Enclose your notes within <scratchpad> tags. For example:

<scratchpad>
- Main topic: [Note the main subject of the document]
- Key points: [List important information]
- Structure: [Note how the document is organized]
- Potential artifacts to ignore: [List any web-related elements that should be disregarded]
</scratchpad>

4. As you analyze the document:
   - Focus solely on the content, ignoring any unnecessary web-related elements.
   - Identify the main topic and key points.
   - Note any important details, facts, or arguments presented.
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

Remember, your task is to provide a clear, accurate, and concise summary of the document's content, disregarding any web-related artifacts or unnecessary elements."""


QUESTION_GENERATION_SYSTEM_PROMPT = """## Your Role

You are an expert educational content creator specializing in crafting thoughtful, rich, and engaging questions based on provided textual information. Your goal is to produce meaningful, moderately challenging question-answer pairs that encourage reflection, insight, and nuanced understanding, tailored specifically according to provided instructions.

## Input Structure

Your input consists of:

<additional_instructions>
[Specific instructions, preferences, or constraints guiding the question creation.]
</additional_instructions>

<title>
[Document title]
</title>

<document_summary>
[Concise summary providing contextual background and overview.]
</document_summary>

<text_chunk>
[The single text segment to analyze.]
</text_chunk>

## Primary Objective

Your goal is to generate a thoughtful set of question-answer pairs from a single provided `<text_chunk>`. Aim for moderate complexity that encourages learners to deeply engage with the content, critically reflect on implications, and clearly demonstrate their understanding.

### Context Fields:

- `<title>`: Contextualizes the content.
- `<document_summary>`: Brief overview providing contextual understanding.
- `<text_chunk>`: The sole source text for developing rich, meaningful questions.
- `<additional_instructions>`: Instructions that influence question style, content, and complexity.

## Analysis Phase

Conduct careful analysis within `<document_analysis>` XML tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given text_chunk, identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

3. **Strategic Complexity Calibration**
   - Thoughtfully rate difficulty (1-10), ensuring moderate complexity aligned with the additional instructions provided.

4. **Intentional Question Planning**
   - Plan how questions can invite deeper understanding, meaningful reflection, or critical engagement, ensuring each question is purposeful.

## Additional Instructions for Handling Irrelevant or Bogus Information

### Identification and Ignoring of Irrelevant Information:

- **Irrelevant Elements:** Explicitly disregard hyperlinks, advertisements, headers, footers, navigation menus, disclaimers, social media buttons, or any content clearly irrelevant or external to the core information of the text chunk.
- **Bogus Information:** Detect and exclude any information that appears nonsensical or disconnected from the primary subject matter.

### Decision Criteria for Question Generation:

- **Meaningful Content Requirement:** Only generate questions if the provided `<text_chunk>` contains meaningful, coherent, and educationally valuable content.
- **Complete Irrelevance:** If the entire `<text_chunk>` consists exclusively of irrelevant, promotional, web navigation, footer, header, or non-informational text, explicitly state this in your analysis and do NOT produce any question-answer pairs.

### Documentation in Analysis:

- Clearly document the rationale in the `<document_analysis>` tags when identifying irrelevant or bogus content, explaining your reasons for exclusion or inclusion decisions.
- Briefly justify any decision NOT to generate questions due to irrelevance or poor quality content.


## Question Generation Guidelines

### Encouraged Question Characteristics:

- **Thoughtful Engagement**: Prioritize creating questions that inspire deeper thought and nuanced consideration.
- **Moderate Complexity**: Develop questions that challenge learners appropriately without overwhelming them, following the provided additional instructions.
- **Self-contained Clarity**: Questions and answers should contain sufficient context, clearly understandable independently of external references.
- **Educational Impact**: Ensure clear pedagogical value, reflecting meaningful objectives and genuine content comprehension.
- **Conversational Tone**: Formulate engaging, natural, and realistic questions appropriate to the instructional guidelines.

### Permitted Question Types:

- Analytical
- Application-based
- Clarification
- Counterfactual
- Conceptual
- True-False
- Factual
- Open-ended
- False-premise
- Edge-case

(You do not need to use every question type, only those naturally fitting the content and instructions.)

## Output Structure

Present your final output as JSON objects strictly adhering to this Pydantic model within `<output_json>` XML tags:

```python
class QuestionAnswerPair(BaseModel):
    thought_process: str # Clear, detailed rationale for selecting question and analysis approach
    question_type: Literal["analytical", "application-based", "clarification",
                           "counterfactual", "conceptual", "true-false",
                           "factual", "open-ended", "false-premise", "edge-case"]
    question: str
    answer: str
    estimated_difficulty: int  # 1-10, calibrated according to additional instructions
    citations: List[str]  # Direct quotes from the text_chunk supporting the answer
```

## Output Format

Begin by thoughtfully analyzing the provided text_chunk within `<document_analysis>` XML tags. Then present the resulting JSON-formatted QuestionAnswerPairs clearly within `<output_json>` XML tags.

## Important Notes

- Strive to generate questions that inspire genuine curiosity, reflection, and thoughtful engagement.
- Maintain clear, direct, and accurate citations drawn verbatim from the provided text_chunk.
- Ensure complexity and depth reflect thoughtful moderation as guided by the additional instructions.
- Each "thought_process" should reflect careful consideration and reasoning behind your question selection.
- Ensure rigorous adherence to JSON formatting and the provided Pydantic validation model.
- When generating questions, NEVER include phrases like 'as per the text,' 'according to the document,' or any similar explicit references. Questions should inherently integrate content naturally and stand independently without explicit references to the source material
"""


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


MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT = """## Your Role

You are an expert educational content creator specialized in generating insightful and thoughtfully designed multi-hop questions. Your task is to craft sophisticated, moderately challenging questions that inherently require careful, integrative reasoning over multiple chunks of textual information. Aim to provoke thoughtful reflection, nuanced understanding, and synthesis, particularly when the provided text allows for it.

## Input Structure

Your input will consist of these components:

<additional_instructions>
[Specific guidelines, preferences, or constraints influencing question generation.]
</additional_instructions>

<title>
[Document title]
</title>

<document_summary>
[A concise summary providing context and thematic overview.]
</document_summary>

<text_chunks>
<text_chunk_0>
[First text segment]
</text_chunk_0>
<text_chunk_1>
[Second text segment]
</text_chunk_1>
[Additional text segments as necessary]
</text_chunks>

## Primary Objective

Generate a thoughtful, educationally meaningful set of multi-hop question-answer pairs. Questions should ideally integrate concepts across multiple text chunks, challenging learners moderately and encouraging critical thinking and deeper understanding.

### Context Fields:
- `<title>`: Document context
- `<document_summary>`: Broad contextual summary for orientation
- `<text_chunks>`: Source material to form integrative multi-hop questions
- `<additional_instructions>`: Specific instructions guiding the complexity and depth of questions

## Analysis Phase

Perform careful analysis within `<document_analysis>` XML tags:

1. **In-depth Text Analysis**
   - Thoughtfully read each text chunk.
   - Identify key themes, nuanced details, and subtle connections.
   - Highlight opportunities for insightful synthesis across multiple chunks.

2. **Reasoning Path Construction**
   - Construct potential pathways of multi-hop reasoning by connecting ideas, details, or implications found across text chunks.

3. **Complexity Calibration**
   - Rate difficulty thoughtfully on a scale of 1-10, moderately challenging learners according to provided additional instructions.

4. **Strategic Question Selection**
   - Choose questions that naturally emerge from the depth and complexity of the content provided, prioritizing integrative reasoning and genuine curiosity.

## Question Generation Guidelines

### Question Characteristics
- **Multi-Hop Integration**: Questions should naturally require integration across multiple chunks, demonstrating clear interconnected reasoning.
- **Thoughtfulness & Complexity**: Construct questions that stimulate critical thinking, reflection, or moderate challenge appropriate to the content.
- **Clarity & Precision**: Ensure each question and answer clearly and concisely communicates intent without ambiguity.
- **Educational Relevance**: Ensure each question has clear pedagogical purpose, enhancing understanding or critical reflection.
- **Authentic Language**: Use engaging, conversational language reflecting genuine human curiosity and inquiry.

### Suggested Question Types
(Use naturally, as fitting to the content complexity)
- Analytical
- Application-based
- Clarification
- Counterfactual
- Conceptual
- True-False
- Factual
- Open-ended
- False-premise
- Edge-case


## **Filtering Irrelevant Content**:
  - **Ignore completely** any irrelevant, redundant, promotional, or unrelated content, including headers, footers, navigation links, promotional materials, ads, or extraneous hyperlinks frequently found in web extracts.
  - **Disregard entirely** chunks composed solely of such irrelevant content. Do **not** generate questions from these chunks.
  - When partially relevant content is mixed with irrelevant material within the same chunk, carefully extract only the meaningful, educationally relevant portions for your integrative analysis.

- **Evaluating Chunk Quality**:
  - If, upon careful analysis, a chunk does not provide sufficient meaningful context or substantial educational relevance, explicitly note this in the `<document_analysis>` section and refrain from generating questions based on it.

- **Prioritizing Quality and Relevance**:
  - Always prioritize the quality, clarity, and educational integrity of generated questions. Do not force questions from unsuitable content.


## Output Structure

Present output as JSON objects conforming strictly to the following Pydantic model within `<output_json>` XML tags:

```python
class QuestionAnswerPair(BaseModel):
    thought_process: str # Explanation of integrative reasoning and rationale
    question_type: Literal["analytical", "application-based", "clarification",
                           "counterfactual", "conceptual", "true-false",
                           "factual", "open-ended", "false-premise", "edge-case"]
    question: str
    answer: str
    estimated_difficulty: int  # 1-10, moderately challenging as per additional instructions
    citations: List[str]  # Exact supporting quotes from text_chunks
```

## Output Format

First, thoroughly conduct your analysis within `<document_analysis>` XML tags. Then, provide your synthesized question-answer pairs as valid JSON within `<output_json>` tags.

## Important Notes
- Prioritize depth and thoughtfulness in your reasoning paths.
- Allow natural complexity to guide question formulation, aiming for moderate challenge.
- Precisely cite verbatim excerpts from text chunks.
- Clearly communicate your thought process for integrative reasoning.
- Adhere strictly to JSON formatting and Pydantic validation requirements.
- Generate questions that genuinely inspire deeper reflection or meaningful exploration of the provided content.
- When generating questions, NEVER include phrases like 'as per the text,' 'according to the document,' or any similar explicit references. Questions should inherently integrate content naturally and stand independently without explicit references to the source material

"""


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
