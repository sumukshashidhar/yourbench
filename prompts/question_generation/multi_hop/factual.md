# Multi-Hop Document-Based Factual Question Generation

You will receive multiple document extracts (chunks) and a summary. Your task is to generate high-quality factual questions that test recall and comprehension of specific information across multiple chunks of the provided text extracts.

## Core Principles

1. **Cross-Chunk Evidence**
   - Questions MUST be answerable with specific words/phrases from multiple chunks
   - Answers must be derived from connecting information across chunks
   - No external knowledge requirements
   - No inference beyond explicit facts

2. **Question Diversity**
   - Mix of entity identification across chunks
   - Temporal relationships between events in different chunks
   - Quantitative facts requiring multiple chunk synthesis
   - Relationship identification spanning chunks

3. **Question Quality**
   - Clear, specific questions with unambiguous answers
   - Answers must be directly present across chunks
   - Questions should target meaningful connections
   - Answers should synthesize information from multiple chunks

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

class ChunkAnalysis(BaseModel):
    chunk_id: str
    content_summary: str
    relevant_information: str
    connection_points: List[str]

class QuestionType(str, Enum):
    FACTUAL = "factual"  # Questions requiring specific fact recall from text

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Simple fact identification (names, dates)
    EASY = 2         # Basic fact recall (events, places)
    MEDIUM = 3       # Multi-part facts (relationships, sequences)
    HARD = 4         # Complex facts (interconnected details)
    VERY_HARD = 5    # Detailed specifics requiring careful reading

[Previous ComprehensionType Enum remains exactly the same]

class QuestionQuality(BaseModel):
    clear_language: bool = Field(..., description="Uses unambiguous language")
    text_based: bool = Field(..., description="Answer directly present in text")
    no_tricks: bool = Field(..., description="Avoids misleading phrasing")

class GeneratedQuestionAnswerPair(BaseModel):
    """
    Represents a structured QA pair for multi-chunk document comprehension testing.
    """
    # Analysis Fields
    document_extract_analysis: str = Field(
        ...,
        description="Analysis of the key points and structure of the extract",
        min_length=50
    )
    
    chunk_analyses: List[ChunkAnalysis] = Field(
        ...,
        description="Analysis of individual chunks and their connections",
        min_items=2
    )
    
    testable_concepts: List[str] = Field(
        ...,
        description="Key concepts that can be tested from the extract",
        min_items=2
    )

    potential_question_directions: List[str] = Field(..., description="The possible questions that a human would likely ask")
    best_direction: str = Field(..., description="The best question to ask, a decision made based on the question_directions. Why would it be a good question to ask, and what skills would it test?")

    # Question Formation Fields
    comprehension_type: ComprehensionType = Field(
        ...,
        description="The type of comprehension being tested"
    )
    
    quality_metrics: QuestionQuality = Field(
        ...,
        description="Quality checks for the question"
    )

    # Evidence Fields
    supporting_quotes: List[str] = Field(
        ...,
        description="Verbatim quotes from text that prove the answer",
        min_items=1
    )
    
    quote_context: str = Field(
        ...,
        description="Explanation of how quotes support the answer",
        min_length=30
    )

    # Core Question Fields
    kind: QuestionType = Field(
        default=QuestionType.FACTUAL,
        description="Question type (factual)"
    )
    
    question: str = Field(
        ...,
        description="The question"
    )
    
    answer: str = Field(
        ...,
        description="The correct answer"
    )
    
    reasoning: str = Field(
        ...,
        description="Detailed explanation of the answer",
        min_length=50
    )
    
    difficulty: DifficultyLevel = Field(
        ...,
        description="Question difficulty level"
    )
    
    difficulty_justification: str = Field(
        ...,
        description="Explanation of difficulty rating",
        min_length=30
    )

    class Config:
        use_enum_values = True
```

## Question Generation Process

1. **Cross-Chunk Fact Identification**
   - Scan multiple chunks for related facts
   - Note entities appearing across chunks
   - Identify temporal relationships between chunks
   - Locate technical terms and definitions that span chunks

2. **Multi-Hop Answer Verification**
   - Ensure answer requires information from multiple chunks
   - Confirm answer can be supported by quotes from different chunks
   - Verify single correct answer exists
   - Check answer synthesizes information correctly

3. **Question Formation**
   - Create clear questions requiring multi-chunk understanding
   - Ensure question targets specific cross-chunk relationships
   - Vary question types across comprehension categories
   - Maintain precise language

4. **Quality Verification**
   - Check answer is supported by multiple chunks
   - Verify no external knowledge needed
   - Confirm clear, unambiguous question
   - Test for answer specificity

## Examples

### Example 1: Cross-Chunk Entity Relationship (Medium)

```json
{
    "document_extract_analysis": "The extracts describe Einstein's work on relativity and its impact on nuclear physics.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Einstein's publication of special relativity in 1905",
            "relevant_information": "Publication date and core concepts of special relativity",
            "connection_points": ["Einstein", "relativity theory", "1905"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Application of E=mc² in nuclear physics",
            "relevant_information": "Connection between mass-energy equivalence and nuclear reactions",
            "connection_points": ["Einstein", "E=mc²", "nuclear physics"]
        }
    ],
    "testable_concepts": [
        "theoretical physics development",
        "scientific applications",
        "historical timeline"
    ],
    "potential_question_directions": [
        "How did Einstein's 1905 theory connect to nuclear physics?",
        "What was the timeline of relativity's application to nuclear science?",
        "How did E=mc² bridge theoretical and applied physics?"
    ],
    "best_direction": "How did Einstein's 1905 theory connect to nuclear physics?",
    "comprehension_type": "relationship",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Einstein published his special theory of relativity in 1905",
        "The equation E=mc² became fundamental to understanding nuclear reactions"
    ],
    "quote_context": "The quotes connect Einstein's 1905 theory to nuclear physics applications through E=mc²",
    "kind": "factual",
    "question": "What equation from Einstein's 1905 theory became fundamental to nuclear physics?",
    "answer": "E=mc²",
    "reasoning": "The text shows that Einstein published special relativity in 1905, which included E=mc², and this equation became crucial for nuclear physics.",
    "difficulty": 3,
    "difficulty_justification": "Requires connecting information across chunks about theory and application."
}
```

### Example 2: Cross-Chunk Temporal Sequence (Hard)

```json
{
    "document_extract_analysis": "The extracts detail the development and testing of the atomic bomb across multiple locations and timeframes.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Manhattan Project initiation and early development",
            "relevant_information": "Project start date and initial research phase",
            "connection_points": ["Manhattan Project", "1942", "research phase"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Trinity test and subsequent events",
            "relevant_information": "Test date and location details",
            "connection_points": ["Trinity test", "July 1945", "Alamogordo"]
        }
    ],
    "testable_concepts": [
        "project timeline",
        "location sequence",
        "development phases"
    ],
    "potential_question_directions": [
        "What was the timeline from project start to first test?",
        "How did locations change throughout the project?",
        "What were the key development milestones?"
    ],
    "best_direction": "What was the timeline from project start to first test?",
    "comprehension_type": "temporal",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "The Manhattan Project began in 1942",
        "The Trinity test was conducted on July 16, 1945"
    ],
    "quote_context": "The quotes establish the start and end points of the initial development phase",
    "kind": "factual",
    "question": "How many years passed between the Manhattan Project's initiation and the Trinity test?",
    "answer": "3 years",
    "reasoning": "By comparing the project's start in 1942 to the Trinity test in July 1945, we can determine the time span.",
    "difficulty": 4,
    "difficulty_justification": "Requires synthesizing dates from multiple chunks and calculating time difference."
}
```

### Example 3: Cross-Chunk Quantitative Analysis (Very Hard)

```json
{
    "document_extract_analysis": "The extracts provide detailed statistics about global climate change across different decades.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "1980s temperature and CO2 levels",
            "relevant_information": "Baseline measurements from 1980s",
            "connection_points": ["temperature", "CO2 levels", "1980s"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "2020s climate measurements",
            "relevant_information": "Current measurements and changes",
            "connection_points": ["temperature increase", "CO2 change", "2020s"]
        }
    ],
    "testable_concepts": [
        "climate change metrics",
        "long-term trends",
        "measurement comparisons"
    ],
    "potential_question_directions": [
        "How have temperature measurements changed?",
        "What is the rate of CO2 increase?",
        "What are the comparative statistics?"
    ],
    "best_direction": "What is the rate of CO2 increase?",
    "comprehension_type": "quantitative",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "CO2 levels in 1980 were 340ppm",
        "By 2020, CO2 levels reached 415ppm"
    ],
    "quote_context": "The quotes provide measurements for calculating the CO2 increase rate",
    "kind": "factual",
    "question": "What was the total increase in CO2 levels from 1980 to 2020?",
    "answer": "75ppm",
    "reasoning": "By subtracting the 1980 level (340ppm) from the 2020 level (415ppm), we can determine the total increase.",
    "difficulty": 5,
    "difficulty_justification": "Requires identifying and calculating change from measurements across multiple chunks."
}
```

## Common Pitfalls to Avoid

1. **Single-Chunk Questions**
   ❌ "What was the CO2 level in 1980?"
   ✅ "How much did CO2 levels increase between 1980 and 2020?"

2. **Inferential Connections**
   ❌ "Why did Einstein's theory lead to nuclear physics?"
   ✅ "Which equation from Einstein's 1905 theory was applied to nuclear physics?"

3. **Unclear Chunk Relationships**
   ❌ "What happened during the Manhattan Project?"
   ✅ "How long did it take from the project's initiation to the first successful test?"

4. **External Knowledge Required**
   ❌ "Why was the CO2 increase significant?"
   ✅ "What was the measured increase in CO2 levels between the two time periods?"

## Output Requirements

1. Generate 3-5 multi-hop factual questions requiring information from multiple chunks
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure at least one question is difficulty level 1 or 2
4. Include at least one question at difficulty level 4 or 5
5. All answers must be supported by quotes from multiple chunks
6. Questions must target specific, verifiable cross-chunk relationships

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Medium/Relationship)
    },
    {
        // Question 2 (Hard/Temporal)
    },
    {
        // Question 3 (Very Hard/Quantitative)
    },
    // ...
]
</generated_questions>
```