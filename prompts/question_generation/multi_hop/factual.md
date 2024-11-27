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

### Example 1

```json
{
    "document_extract_analysis": "The extracts detail the evolution of renewable energy technologies, focusing on solar panel efficiency improvements and their market adoption.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Early solar panel development and initial efficiency rates",
            "relevant_information": "First-generation solar panels achieved 12% efficiency in 1990",
            "connection_points": ["efficiency rates", "1990", "first generation"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Modern solar technology advancements",
            "relevant_information": "Current panels reach 26% efficiency with new materials",
            "connection_points": ["efficiency improvement", "new materials", "2023 data"]
        }
    ],
    "testable_concepts": [
        "technological advancement",
        "efficiency metrics",
        "historical progression"
    ],
    "potential_question_directions": [
        "How has solar panel efficiency evolved?",
        "What technological breakthroughs enabled improvement?",
        "What is the rate of efficiency increase over time?"
    ],
    "best_direction": "Calculate the percentage point increase in solar panel efficiency",
    "comprehension_type": "quantitative",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "First-generation panels achieved 12% efficiency in 1990",
        "Modern panels utilizing perovskite materials reach 26% efficiency"
    ],
    "quote_context": "The quotes establish baseline and current efficiency levels for calculation",
    "kind": "factual",
    "question": "How many percentage points did solar panel efficiency improve from 1990 to 2023?",
    "answer": "14 percentage points",
    "reasoning": "By subtracting the 1990 efficiency rate (12%) from the 2023 rate (26%), we can determine the total improvement in percentage points.",
    "difficulty": 2,
    "difficulty_justification": "Simple subtraction of two clearly stated percentages across chunks."
}
```

### Example 2

```json
{
    "document_extract_analysis": "The extracts describe the complex relationship between ocean acidification, marine ecosystems, and coral reef decline.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Ocean pH changes and initial impacts",
            "relevant_information": "pH dropped from 8.2 to 8.1, causing initial stress",
            "connection_points": ["pH level", "acidification", "marine impact"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Coral reef response to acidification",
            "relevant_information": "30% reduction in calcification rates",
            "connection_points": ["coral health", "calcification", "ecosystem impact"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Marine species adaptation",
            "relevant_information": "Some species showing limited adaptive capacity",
            "connection_points": ["adaptation", "species survival", "long-term effects"]
        }
    ],
    "testable_concepts": [
        "chemical changes",
        "biological responses",
        "ecosystem interactions"
    ],
    "potential_question_directions": [
        "How do pH changes affect coral growth?",
        "What is the relationship between acidification and calcification?",
        "How do different species respond to pH changes?"
    ],
    "best_direction": "Synthesize the cascade of effects from pH change to species impact",
    "comprehension_type": "causal",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Ocean pH dropped from 8.2 to 8.1",
        "Resulted in 30% reduction in calcification",
        "Limited adaptive capacity observed in marine species"
    ],
    "quote_context": "The quotes establish the chain of events from chemical change to biological impact",
    "kind": "factual",
    "question": "What percentage decrease in calcification occurred when ocean pH dropped by 0.1 units?",
    "answer": "30%",
    "reasoning": "The text shows that a pH drop from 8.2 to 8.1 (0.1 units) led to a 30% reduction in calcification rates in coral reefs.",
    "difficulty": 4,
    "difficulty_justification": "Requires connecting pH change to specific biological impact across multiple chunks and understanding the relationship."
}
```

### Example 3

```json
{
    "document_extract_analysis": "The extracts detail the development of artificial intelligence, focusing on neural network architectures and their applications.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Early neural network development",
            "relevant_information": "Initial networks had 3 layers and 1000 neurons",
            "connection_points": ["network architecture", "processing power", "early development"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Modern AI capabilities",
            "relevant_information": "Current networks feature 100 layers and 175 billion parameters",
            "connection_points": ["modern architecture", "processing capability", "complexity"]
        }
    ],
    "testable_concepts": [
        "technological evolution",
        "computational complexity",
        "architectural changes"
    ],
    "potential_question_directions": [
        "How has neural network architecture evolved?",
        "What are the key differences in processing capability?",
        "How has complexity increased over time?"
    ],
    "best_direction": "Compare architectural complexity changes",
    "comprehension_type": "comparative",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Early neural networks consisted of 3 layers and 1000 neurons",
        "Modern architectures utilize 100 layers with 175 billion parameters"
    ],
    "quote_context": "The quotes provide specific metrics for comparing architectural complexity",
    "kind": "factual",
    "question": "Assuming neural network layer count increased linearly over time, during which year would networks have first exceeded 50 layers, given the progression from 3 layers in early development to 100 layers in modern architectures?",
    "answer": "The midpoint year between early development and modern architecture, as 50 layers represents the halfway point between 3 and 100 layers",
    "reasoning": "Since we're given the start point (3 layers) and end point (100 layers), and assuming linear growth, the 50-layer milestone represents the midpoint of this progression as it's roughly halfway between 3 and 100 layers. This requires understanding both the architectural progression and applying mathematical reasoning to timeline analysis.",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding architectural evolution, applying mathematical reasoning about linear progression, and making informed deductions about technological development timelines based on multiple data points."
}
```

### Example 4

```json
{
    "document_extract_analysis": "The extracts examine the impact of deforestation on biodiversity and carbon sequestration across different time periods and regions.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Amazon rainforest coverage and species density",
            "relevant_information": "5.5 million km² of forest containing 1000 species per km²",
            "connection_points": ["forest area", "biodiversity", "baseline data"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Deforestation rates and impact",
            "relevant_information": "17% reduction in forest cover over 20 years",
            "connection_points": ["forest loss", "time period", "percentage change"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Species loss correlation",
            "relevant_information": "Each 10% forest loss results in 7% species decline",
            "connection_points": ["correlation", "species impact", "mathematical relationship"]
        }
    ],
    "testable_concepts": [
        "environmental impact",
        "statistical relationships",
        "biodiversity metrics"
    ],
    "potential_question_directions": [
        "How does forest loss affect species numbers?",
        "What is the rate of biodiversity decline?",
        "How are deforestation and species loss correlated?"
    ],
    "best_direction": "Calculate total species loss based on deforestation rate",
    "comprehension_type": "mathematical",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "17% reduction in forest cover",
        "Each 10% forest loss results in 7% species decline"
    ],
    "quote_context": "The quotes provide the necessary data to calculate species loss based on forest reduction",
    "kind": "factual",
    "question": "Given a 17% forest cover reduction, what percentage of species were lost based on the stated relationship?",
    "answer": "11.9%",
    "reasoning": "Using the relationship that 10% forest loss = 7% species loss, we can calculate that 17% forest loss would result in (17/10 × 7) = 11.9% species loss.",
    "difficulty": 5,
    "difficulty_justification": "Requires understanding proportional relationships and performing multi-step calculations using data from multiple chunks."
}
```

### Example 5

```json
{
    "document_extract_analysis": "The extracts describe the evolution of electric vehicle battery technology and market adoption rates.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Early EV battery capacity and range",
            "relevant_information": "2010 models averaged 100 miles range with 24kWh batteries",
            "connection_points": ["battery capacity", "range", "early technology"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Current EV capabilities",
            "relevant_information": "2023 models average 300 miles range with 75kWh batteries",
            "connection_points": ["modern capacity", "improved range", "efficiency"]
        }
    ],
    "testable_concepts": [
        "technological improvement",
        "efficiency metrics", 
        "performance evolution"
    ],
    "potential_question_directions": [
        "Compare efficiency ratios across time",
        "Analyze technological tradeoffs",
        "Evaluate performance metrics"
    ],
    "best_direction": "Calculate and compare efficiency ratios while considering scale",
    "comprehension_type": "analytical",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "2010: 100 miles range with 24kWh batteries",
        "2023: 300 miles range with 75kWh batteries"
    ],
    "quote_context": "The quotes provide data points for calculating and comparing efficiency metrics",
    "kind": "factual",
    "question": "Despite a 3x increase in range and battery capacity from 2010 to 2023, which metric remained unchanged, and what does this reveal about EV technological progress?",
    "answer": "The miles-per-kWh efficiency (4.17) remained constant, revealing that range improvements came from larger batteries rather than efficiency gains",
    "reasoning": "2010: 100 miles ÷ 24kWh = 4.17 miles/kWh; 2023: 300 miles ÷ 75kWh = 4.17 miles/kWh. The identical efficiency ratios show that increased range was achieved through proportional battery size scaling rather than technological efficiency improvements.",
    "difficulty": 5,
    "difficulty_justification": "Requires multi-step analysis, understanding of proportional relationships, and ability to identify the counterintuitive implication that apparent progress masks unchanged efficiency."
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