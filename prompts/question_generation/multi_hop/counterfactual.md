# Multi-Document Counterfactual Question Generation

You will receive multiple document extracts (chunks) and a summary. Your task is to generate high-quality counterfactual questions that explore alternate scenarios based on changing specific elements from the provided text extracts, requiring information from multiple chunks to answer.

## Core Principles

1. **Multi-Chunk Integration**
   - Questions must require information from multiple chunks
   - Changes must affect relationships across chunks
   - Answers must synthesize information
   - Reasoning must connect multiple pieces

2. **Text-Based Alterations**
   - Each question must modify specific facts/events from the texts
   - Answers must follow logically from cross-chunk relationships
   - Changes must be clearly defined
   - Reasoning must use evidence from multiple chunks

3. **Question Diversity**
   - Event modifications affecting multiple chunks
   - Character/entity changes with cross-chunk impact
   - Decision alterations with cascading effects
   - Timing variations across chunks
   - Condition changes affecting multiple sections
   - Process modifications with broad impact

4. **Question Quality**
   - Clear modification of text elements
   - Logical chain of consequences across chunks
   - Grounded in multi-chunk relationships
   - Plausible alternative scenarios

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
    COUNTERFACTUAL = "counterfactual"  # Questions exploring alternative scenarios

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Simple direct change
    EASY = 2         # Basic alternative scenario
    MEDIUM = 3       # Multi-step consequences
    HARD = 4         # Complex chain of effects
    VERY_HARD = 5    # System-wide implications

[... rest of the enums remain exactly the same as in original prompt ...]

class QuestionQuality(BaseModel):
    clear_language: bool = Field(..., description="Uses unambiguous language")
    text_based: bool = Field(..., description="Changes based on text elements")
    no_tricks: bool = Field(..., description="Avoids unrealistic scenarios")

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
        description="Key concepts that can be tested from the extracts",
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
        default=QuestionType.COUNTERFACTUAL,
        description="Question type (counterfactual)"
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

1. **Cross-Chunk Analysis**
   - Identify relationships between chunks
   - Map information dependencies
   - Note shared elements
   - Track narrative connections

2. **Element Identification**
   - Identify key changeable elements across chunks
   - Note critical decision points affecting multiple sections
   - Map cross-chunk causal relationships
   - Identify system dependencies spanning chunks

3. **Change Mapping**
   - Define clear modifications with multi-chunk impact
   - Trace impact chains across chunks
   - Consider system effects across document
   - Evaluate plausibility of cross-chunk changes

4. **Question Formation**
   - Specify clear changes affecting multiple chunks
   - Follow logical consequences across sections
   - Maintain text relationships between chunks
   - Consider scope of multi-chunk impact

5. **Quality Verification**
   - Check change clarity across chunks
   - Verify logical flow between sections
   - Confirm text basis from multiple chunks
   - Test plausibility of cross-chunk effects

## Examples

### Example 1: Cross-Chunk Event Change (Medium)

```json
{
    "document_extract_analysis": "The text describes the development and impact of the Manhattan Project across multiple locations and phases.",
    "chunk_analyses": [
        {
            "chunk_id": "MP_1",
            "content_summary": "Details the initial research phase at University of Chicago",
            "relevant_information": "Fermi's team achieved first controlled nuclear reaction",
            "connection_points": ["leads to Los Alamos work", "proves theoretical possibility"]
        },
        {
            "chunk_id": "MP_2",
            "content_summary": "Describes the Los Alamos laboratory establishment",
            "relevant_information": "Oppenheimer led weapon development based on Chicago findings",
            "connection_points": ["builds on Chicago research", "enables Trinity test"]
        }
    ],
    "testable_concepts": [
        "scientific collaboration",
        "research progression",
        "project management"
    ],
    "potential_question_directions": [
        "What if Fermi's Chicago experiment had failed to achieve a controlled reaction?",
        "How would a delay in the Chicago phase have affected Los Alamos work?",
        "What if the research findings couldn't be transferred between locations?"
    ],
    "best_direction": "What if Fermi's Chicago experiment had failed to achieve a controlled reaction?",
    "comprehension_type": "event_change",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Fermi's successful experiment proved the theoretical calculations correct",
        "Los Alamos work proceeded based on the Chicago team's findings",
        "Oppenheimer's team built upon the established nuclear reaction principles"
    ],
    "quote_context": "The quotes show the dependency between Chicago's success and Los Alamos development.",
    "kind": "counterfactual",
    "question": "Based on the text's description of the Manhattan Project's phases, what would have been the likely impact on the Los Alamos laboratory's work if Fermi's team had failed to achieve a controlled nuclear reaction in Chicago?",
    "answer": "The Los Alamos weapon development program would have lacked the experimental validation needed to proceed, likely causing significant delays or a fundamental project redesign",
    "reasoning": "The text shows that Los Alamos work was built directly on Chicago's success. Without experimental proof of concept, the theoretical basis for the Los Alamos work would have been uncertain, requiring either additional basic research or a different approach.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding the relationship between research phases and tracing consequences across multiple project stages."
}
```

### Example 2: Cross-Chunk Process Change (Hard)

```json
{
    "document_extract_analysis": "The text describes the Apollo 11 mission's multiple stages from launch to lunar landing.",
    "chunk_analyses": [
        {
            "chunk_id": "A11_1",
            "content_summary": "Details Earth orbit and Trans-Lunar Injection",
            "relevant_information": "Critical velocity calculations for lunar trajectory",
            "connection_points": ["affects lunar approach", "determines fuel usage"]
        },
        {
            "chunk_id": "A11_2",
            "content_summary": "Describes lunar orbit insertion process",
            "relevant_information": "Precise timing and velocity requirements",
            "connection_points": ["depends on Earth departure", "enables landing"]
        }
    ],
    "testable_concepts": [
        "orbital mechanics",
        "mission planning",
        "system integration"
    ],
    "potential_question_directions": [
        "What if the Trans-Lunar Injection velocity was significantly lower?",
        "How would different Earth orbit parameters affect lunar approach?",
        "What if orbital calculations used different reference points?"
    ],
    "best_direction": "What if the Trans-Lunar Injection velocity was significantly lower?",
    "comprehension_type": "process_change",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "TLI velocity determined the entire lunar trajectory",
        "Lunar orbit insertion required precise approach velocity",
        "Fuel reserves were calculated based on planned velocities"
    ],
    "quote_context": "The quotes establish the interconnected nature of mission velocities and fuel usage.",
    "kind": "counterfactual",
    "question": "According to the text's description of the Apollo 11 mission phases, what would have been the consequences if the Trans-Lunar Injection velocity had been 10% lower than planned?",
    "answer": "The spacecraft would have failed to reach the Moon, requiring more fuel for correction or mission abort, as the lower velocity would have altered the entire trajectory and orbital insertion calculations",
    "reasoning": "The text shows that TLI velocity determined the trajectory and subsequent fuel usage. A lower velocity would have changed the entire flight path, requiring more fuel for corrections and making the planned lunar orbit insertion impossible with existing fuel reserves.",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding complex relationships between velocity, trajectory, and fuel usage across multiple mission phases."
}
```

### Example 3: Cross-Chunk System Impact (Very Hard)

```json
{
    "document_extract_analysis": "The text explains the global climate system's response to volcanic eruptions.",
    "chunk_analyses": [
        {
            "chunk_id": "VOL_1",
            "content_summary": "Describes initial atmospheric effects of eruptions",
            "relevant_information": "Sulfur dioxide conversion to sulfate aerosols",
            "connection_points": ["affects temperature", "influences circulation"]
        },
        {
            "chunk_id": "VOL_2",
            "content_summary": "Details ocean temperature response",
            "relevant_information": "Ocean circulation changes from cooling",
            "connection_points": ["responds to atmosphere", "affects weather"]
        }
    ],
    "testable_concepts": [
        "atmospheric chemistry",
        "ocean-atmosphere coupling",
        "climate feedback loops"
    ],
    "potential_question_directions": [
        "What if volcanic sulfur dioxide produced warming instead of cooling?",
        "How would different aerosol behavior affect ocean responses?",
        "What if atmospheric circulation patterns were reversed?"
    ],
    "best_direction": "What if volcanic sulfur dioxide produced warming instead of cooling?",
    "comprehension_type": "system_impact",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Sulfate aerosols reflect sunlight, cooling the atmosphere",
        "Ocean circulation responds to atmospheric temperature changes",
        "Weather patterns shift based on temperature gradients"
    ],
    "quote_context": "The quotes establish the chain of effects from atmospheric chemistry to ocean response.",
    "kind": "counterfactual",
    "question": "Based on the text's description of volcanic climate effects, how would the global climate system respond if volcanic sulfur dioxide caused atmospheric warming instead of cooling?",
    "answer": "The ocean circulation would reverse its response, leading to amplified warming and fundamentally different weather patterns, as the temperature gradient-driven processes would operate in opposite directions",
    "reasoning": "The text shows that ocean circulation and weather patterns are driven by atmospheric temperature changes. Reversing the initial temperature effect would cascade through the entire system, creating opposite responses in ocean circulation and atmospheric patterns.",
    "difficulty": 5,
    "difficulty_justification": "Requires understanding complex interactions between atmosphere and ocean systems, and tracing multiple levels of reversed effects."
}
```

## Common Pitfalls to Avoid

1. **Single-Chunk Focus**
   ❌ "What if just the Chicago experiment changed?"
   ✅ "How would changes in Chicago affect Los Alamos?"

2. **Disconnected Changes**
   ❌ "What if unrelated elements changed in each chunk?"
   ✅ "What if a change in one chunk affected related elements in others?"

3. **Missing Connections**
   ❌ "What if something changed without considering cross-chunk impact?"
   ✅ "How would changes propagate across connected elements?"

4. **Unrealistic Changes**
   ❌ "What if physics worked differently in each location?"
   ✅ "What if communication between locations was delayed?"

## Output Requirements

1. Generate 3-5 counterfactual questions requiring multiple chunks
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure changes affect multiple chunks
4. Include at least one system-level change
5. All reasoning must use cross-chunk relationships
6. Changes must be plausible and meaningful

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Medium/Cross-Chunk Event)
    },
    {
        // Question 2 (Hard/Cross-Chunk Process)
    },
    {
        // Question 3 (Very Hard/Cross-Chunk System)
    },
    // ...
]
</generated_questions>
```

## Additional Guidelines

1. **Cross-Chunk Integration**
   - Identify connecting elements
   - Track information flow
   - Map dependencies
   - Note shared concepts

2. **Change Selection**
   - Choose elements affecting multiple chunks
   - Ensure clear modification paths
   - Consider cross-chunk impact
   - Maintain plausibility

3. **Consequence Tracing**
   - Follow effects across chunks
   - Use multi-chunk relationships
   - Consider cascading effects
   - Maintain system consistency

4. **Difficulty Scaling**
   - Simple cross-chunk changes (Level 2-3)
   - Complex dependencies (Level 3-4)
   - System-wide impacts (Level 4-5)
   - Multiple interaction paths (Level 5)

5. **Counterfactual Types**
   - Event: Changed occurrences affecting multiple chunks
   - Decision: Alternative choices with cross-chunk impact
   - Timing: Different sequences across chunks
   - Condition: Modified circumstances affecting multiple sections
   - Process: Alternative methods spanning chunks
   - System: Broad changes affecting entire document

6. **Response Evaluation**
   - Clear cross-chunk change specification
   - Logical consequence chains between chunks
   - Text-based reasoning from multiple sources
   - Plausible outcomes across document