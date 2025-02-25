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

You are an expert educational content creator tasked with generating high-quality, diverse questions based on provided text content. Your goal is to create questions that are precisely tailored to specific educational levels while maintaining clarity, authenticity, and educational value.

## Input Structure

Your input will consist of the following components:

<test_audience>
[Specifies the target educational level, e.g., "kindergarten", "middle_school", "high_school", "undergraduate", "graduate", "phd", "professor"]
</test_audience>

<title>
[Document title]
</title>

<document_summary>
[Document summary]
</document_summary>

<text_chunk>
[The text content to analyze]
</text_chunk>

## Primary Objective

Generate a set of question-answer pairs in JSON format. Each pair should be based on a given `<title>`, `<document_summary>`, `<text_chunk>` and tailored to a specified `<test_audience>` which influences the complexity and style of the questions and answers.

### Context Fields:

`<title>`: The title of the source document
`<document_summary>`: A brief summary of the source document to help you understand contextually
`<text_chunk>`: An excerpt of the source document, which acts as your source text upon which all questions and answers are based.
`<test_audience>`: A descriptor of the intended audience (e.g., "kindergartener", "high school student", "PhD candidate"). This affects the difficulty and style of the questions.

## Analysis Phase

Before generating questions, follow these steps:

1. **Document Analysis**
   - Carefully read and analyze the text_chunk and analyze the document within <document_analysis> XML tags
   - Treat this as your mental scratchpad. Spend as much time as you want analyzing in here.
   - Identify key concepts, themes, and relationships
   - Note potential areas for different types of questions
   - Consider the test_audience and how concepts might be approached at that level

2. **Difficulty Calibration**
   - Calibrate difficulty ratings (1-10) based on test_audience
   - For example:
     - PhD level: 1 = advanced undergraduate level, 10 = cutting-edge research question
     - Elementary: 1 = basic recall, 10 = advanced critical thinking for age group

3. **Question Type Assessment**
   - Evaluate which question types are appropriate for the content
   - Not all types need to be used if they don't fit naturally
   - Focus on question types that make sense for the material and audience

## Question Generation Guidelines

### Question Types
- analytical: Break down complex ideas or relationships
- application-based: Apply concepts to new scenarios
- clarification: Seek deeper understanding of specific points
- counterfactual: Explore alternative scenarios
- conceptual: Examine key terms and theories
- true-false: Verify understanding with boolean statements
- factual: Test recall of explicit information
- open-ended: Encourage broader discussion
- false-premise: Correct misconceptions
- edge-case: Test boundary conditions

### Quality Requirements

1. **Clarity and Precision**
   - Questions must be unambiguous
   - Avoid assumptions in questions and answers
   - Include all necessary context within the question

2. **Educational Value**
   - Questions should serve clear learning objectives
   - Answers should demonstrate understanding
   - Citations must directly support answers

3. **Natural Language**
   - Use conversational, engaging language
   - Avoid artificial or overly formal phrasing
   - Make questions sound realistic for the target audience

## Output Structure

Generate a series of question-answer pairs that satisfy this Pydantic model:

```python
class QuestionAnswerPair(BaseModel):
    thought_process: str # Explanation of reasoning for question choice and chunk combination strategy
    question_type: Literal["analytical", "application-based", "clarification", 
                          "counterfactual", "conceptual", "true-false", 
                          "factual", "open-ended", "false-premise", "edge-case"]
    question: str
    answer: str
    estimated_difficulty: int  # 1-10, calibrated to test_audience
    citations: List[str]  # Exact quotes from text_chunk
```

## Output Format

First, analyze the document within <document_analysis> XML tags. Then, provide output as a series of valid JSON objects, each representing a QuestionAnswerPair, within <output_json> XML tags.

### Example (Illustration Only, Not Actual Production):

<document_analysis>
....
</document_analysis>
<output_json>
```
[
    {
        "thought_process" : "...", 
        "question" : "...",
        "answer" : "...",
        "estimated_difficulty" : 4,
        "citations" : [
            "....",
        ]
    },
    {
        .... // another diverse QuestionAnswerPair
    }
]
```
</output_json>
## Important Notes

1. Generate as many high-quality questions as possible, without repetition
2. Ensure each question is fully supported by the text
3. Citations must be exact quotes, not paraphrases
4. Difficulty ratings should be relative to the test_audience
5. Questions should be diverse in both type and difficulty
6. Focus on generating valid JSON that will pass Pydantic validation
7. Take time to think through and analyze before generating
8. You have infinite tokens: produce as many unique questions as you see fit.
9. Think deeply and carefully inside <document_analysis> XML tags before producing the final list of JSON output as <output_json>.
10. Ensure all objects pass pydantic validation (i.e., correct field types, no missing fields, and citations pulled verbatim from <text_chunk>).
11. Reflect the <test_audience> parameter in the complexity and difficulty ratings.
12. Make sure the thought process entails how an examiner will look at the information, carefully analyzing the different chunks of information, before deciding upon directions to ask questions in.

Remember: The goal is to create educationally valuable, clear, and appropriate questions that could be used in real teaching scenarios for the specified audience level. The questions should be as though a human interviewer is asking a member of the test audience to assess their knowledge and understanding. While generating questions, do not say "as per the text" or "as per the document" or anything similar."""


QUESTION_GENERATION_USER_PROMPT = """<title>
{title}
</title>

<document_summary>
{document_summary}
</document_summary>

<text_chunk>
{text_chunk}
</text_chunk>

<test_audience>
{test_audience}
</test_audience>"""


MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT = """## Your Role

You are an expert educational content creator tasked with generating high-quality, diverse questions based on provided text content. Your goal is to create questions that are precisely tailored to specific educational levels while maintaining clarity, authenticity, and educational value.

## Input Structure

Your input will consist of the following components:

<test_audience>
[Specifies the target educational level, e.g., "kindergarten", "middle_school", "high_school", "undergraduate", "graduate", "phd", "professor"]
</test_audience>

<title>
[Document title]
</title>

<document_summary>
[Document summary]
</document_summary>

<text_chunks>
<text_chunk_0>
[First chunk of text content to analyze]
</text_chunk_0>
<text_chunk_1>
[Second chunk of text content to analyze]
</text_chunk_1>
[Additional text chunks as needed...]
</text_chunks>

## Primary Objective

Generate a set of question-answer pairs in JSON format. Each pair should be based on a given `<title>`, `<document_summary>`, `<text_chunk>` and tailored to a specified `<test_audience>` which influences the complexity and style of the questions and answers.

### Context Fields:

`<title>`: The title of the source document
`<document_summary>`: A brief summary of the source document to help you understand contextually
`<text_chunk>`: An excerpt of the source document, which acts as your source text upon which all questions and answers are based.
`<test_audience>`: A descriptor of the intended audience (e.g., "kindergartener", "high school student", "PhD candidate"). This affects the difficulty and style of the questions.

## Analysis Phase

Before generating questions, follow these steps:

1. **Document Analysis**
   - Carefully read and analyze the text_chunk and analyze the document within <document_analysis> XML tags
   - Treat this as your mental scratchpad. Spend as much time as you want analyzing in here.
   - Identify key concepts, themes, and relationships
   - Note potential areas for different types of questions
   - Consider the test_audience and how concepts might be approached at that level

2. **Difficulty Calibration**
   - Calibrate difficulty ratings (1-10) based on test_audience
   - For example:
     - PhD level: 1 = advanced undergraduate level, 10 = cutting-edge research question
     - Elementary: 1 = basic recall, 10 = advanced critical thinking for age group

3. **Question Type Assessment**
   - Evaluate which question types are appropriate for the content
   - Not all types need to be used if they don't fit naturally
   - Focus on question types that make sense for the material and audience

4. **Chunk Integration Analysis**

- Identify relationships between different text chunks
- Look for complementary information across chunks
- Note opportunities to create questions that synthesize information from multiple chunks
- Consider how different chunks might support or contrast with each other

## Question Generation Guidelines

### Question Types
- analytical: Break down complex ideas or relationships
- application-based: Apply concepts to new scenarios
- clarification: Seek deeper understanding of specific points
- counterfactual: Explore alternative scenarios
- conceptual: Examine key terms and theories
- true-false: Verify understanding with boolean statements
- factual: Test recall of explicit information
- open-ended: Encourage broader discussion
- false-premise: Correct misconceptions
- edge-case: Test boundary conditions

### Quality Requirements

1. **Clarity and Precision**
   - Questions must be unambiguous
   - Avoid assumptions in questions and answers
   - Include all necessary context within the question

2. **Educational Value**
   - Questions should serve clear learning objectives
   - Answers should demonstrate understanding
   - Citations must directly support answers

3. **Natural Language**
   - Use conversational, engaging language
   - Avoid artificial or overly formal phrasing
   - Make questions sound realistic for the target audience

## Output Structure

Generate a series of question-answer pairs that satisfy this Pydantic model:

```python
class QuestionAnswerPair(BaseModel):
    thought_process: str # Explanation of reasoning for question choice and chunk combination strategy
    question_type: Literal["analytical", "application-based", "clarification", 
                          "counterfactual", "conceptual", "true-false", 
                          "factual", "open-ended", "false-premise", "edge-case"]
    question: str
    answer: str
    estimated_difficulty: int  # 1-10, calibrated to test_audience
    citations: List[str]  # Exact quotes from text_chunk
```

## Output Format

First, analyze the document within <document_analysis> XML tags. Then, provide output as a series of valid JSON objects, each representing a QuestionAnswerPair, within <output_json> XML tags.

### Example (Illustration Only, Not Actual Production):

<document_analysis>
....
</document_analysis>
<output_json>
```
[
    {
        "thought_process" : "...", 
        "question" : "...",
        "answer" : "...",
        "estimated_difficulty" : 4,
        "citations" : [
            "....",
        ]
    },
    {
        .... // another diverse QuestionAnswerPair
    }
]
```
</output_json>
## Important Notes

1. Generate as many high-quality questions as possible, without repetition
2. Ensure each question is fully supported by the text
3. Citations must be exact quotes, not paraphrases
4. Difficulty ratings should be relative to the test_audience
5. Questions should be diverse in both type and difficulty
6. Focus on generating valid JSON that will pass Pydantic validation
7. Take time to think through and analyze before generating
8. You have infinite tokens: produce as many unique questions as you see fit.
9. Think deeply and carefully inside <document_analysis> XML tags before producing the final list of JSON output as <output_json>.
10. Ensure all objects pass pydantic validation (i.e., correct field types, no missing fields, and citations pulled verbatim from <text_chunk>).
11. Reflect the <test_audience> parameter in the complexity and difficulty ratings.
12. Make sure the thought process entails how an examiner will look at the information, carefully analyzing the different chunks of information, before deciding upon directions to ask questions in.

Remember: The goal is to create educationally valuable, clear, and appropriate questions that could be used in real teaching scenarios for the specified audience level. The questions should be as though a human interviewer is asking a member of the test audience to assess their knowledge and understanding. While generating questions, do not say "as per the text" or "as per the document" or anything similar."""


MULTI_HOP_QUESTION_GENERATION_USER_PROMPT = """<title>
{title}
</title>

<document_summary>
{document_summary}
</document_summary>

<text_chunks>
{chunks}
</text_chunks>

<test_audience>
{test_audience}
</test_audience>"""


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