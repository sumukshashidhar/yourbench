# Multi-Document False Premise Question Generation

You will receive multiple document extracts from the same source document and a summary. Your task is to generate high-quality false premise questions that present incorrect assumptions or assertions that contradict information across the provided text extracts, requiring multi-hop reasoning.

## Core Principles

1. **Cross-Text Contradiction**
   - False premises must contradict multiple text chunks
   - Correction must be supported by multiple chunks
   - No external knowledge required
   - Clear evidence for correction from multiple sources

2. **Question Diversity**
   - Multi-hop factual contradictions
   - Cross-chunk process inversions
   - Inter-chunk relationship errors
   - Complex temporal mistakes
   - Multi-step causal errors
   - Cross-reference definition mistakes

3. **Question Quality**
   - Clear false premise spanning multiple chunks
   - Obvious text contradictions across chunks
   - Evidence-based correction from multiple sources
   - Meaningful error requiring multi-hop reasoning

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
    FALSE_PREMISE = "false-premise"  # Questions presenting incorrect assumptions for correction

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Simple fact contradiction
    EASY = 2         # Basic relationship error
    MEDIUM = 3       # Process inversion
    HARD = 4         # Complex relationship error
    VERY_HARD = 5    # System-level contradiction

class ComprehensionType(str, Enum):
    FACT_RECALL = "fact_recall"           # Basic facts from the text
    RELATIONSHIP = "relationship"          # Connections between ideas
    MAIN_IDEA = "main_idea"               # Central concepts
    DETAIL = "detail"                     # Specific details
    SEQUENCE = "sequence"                 # Order of events/ideas
    CAUSE_EFFECT = "cause_effect"         # Causal relationships
    ENTITY = "entity"                 # People, places, things
    TEMPORAL = "temporal"             # Dates, times, sequences
    QUANTITATIVE = "quantitative"     # Numbers, amounts, measures
    LOCATION = "location"             # Places, positions, settings
    ATTRIBUTE = "attribute"           # Characteristics, properties
    ACTION = "action"                 # Events, activities, behaviors
    TECHNICAL = "technical"           # Specialized terms, concepts
    COMPARE_CONTRAST = "compare_contrast"     # Similarities and differences
    PROCESS_ANALYSIS = "process_analysis"      # How things work/happen
    PATTERN_RECOGNITION = "pattern_recognition" # Identifying trends/patterns
    ARGUMENT_EVALUATION = "argument_evaluation" # Assessing claims/evidence
    SYSTEM_ANALYSIS = "system_analysis"        # Understanding whole systems
    EVIDENCE_SYNTHESIS = "evidence_synthesis"   # Combining multiple pieces
    IMPLICATION_ANALYSIS = "implication_analysis" # Understanding consequences
    EVENT_CHANGE = "event_change"           # Modifying key events
    DECISION_CHANGE = "decision_change"     # Alternative choices
    TIMING_CHANGE = "timing_change"         # Different temporal scenarios
    CONDITION_CHANGE = "condition_change"   # Modified circumstances
    ENTITY_CHANGE = "entity_change"         # Different actors/elements
    PROCESS_CHANGE = "process_change"       # Alternative methods
    CHAIN_EFFECT = "chain_effect"          # Multiple linked changes
    SYSTEM_IMPACT = "system_impact"        # Broad systemic effects
    PRINCIPLE_UNDERSTANDING = "principle_understanding"  # Core ideas/rules
    CONCEPT_APPLICATION = "concept_application"         # Using concepts
    RELATIONSHIP_COMPREHENSION = "relationship_comprehension"  # Links between ideas
    FRAMEWORK_ANALYSIS = "framework_analysis"          # Understanding structures
    THEORY_EXPLANATION = "theory_explanation"         # Theoretical foundations
    MODEL_COMPREHENSION = "model_comprehension"      # Understanding models
    MECHANISM_UNDERSTANDING = "mechanism_understanding"  # How things work
    CONCEPT_SYNTHESIS = "concept_synthesis"         # Combining ideas
    TERM_DEFINITION = "term_definition"           # Unclear terminology
    PROCESS_EXPLANATION = "process_explanation"   # How things work
    RELATIONSHIP_CLARITY = "relationship_clarity" # Connections between elements
    CONTEXT_CLARIFICATION = "context_clarification" # Background needed
    REFERENCE_RESOLUTION = "reference_resolution"  # Unclear references
    DETAIL_EXPLANATION = "detail_explanation"     # Specific points
    AMBIGUITY_RESOLUTION = "ambiguity_resolution" # Unclear meanings
    TECHNICAL_CLARIFICATION = "technical_clarification" # Complex concepts
    EXCEPTION_CASE = "exception_case"           # Rule exceptions
    BOUNDARY_CONDITION = "boundary_condition"   # Limit cases
    SPECIAL_CIRCUMSTANCE = "special_circumstance" # Unusual conditions
    RULE_LIMITATION = "rule_limitation"         # Where rules fail
    EXTREME_CASE = "extreme_case"              # Extreme conditions
    CORNER_CASE = "corner_case"                # Unusual combinations
    SYSTEM_BOUNDARY = "system_boundary"        # System limits
    CONDITION_INTERACTION = "condition_interaction" # Multiple factors
    FACT_CONTRADICTION = "fact_contradiction"     # Wrong facts
    PROCESS_INVERSION = "process_inversion"       # Incorrect sequences
    RELATIONSHIP_ERROR = "relationship_error"     # Wrong connections
    TEMPORAL_MISTAKE = "temporal_mistake"         # Time sequence errors
    CAUSAL_ERROR = "causal_error"               # Wrong cause-effect
    DEFINITION_ERROR = "definition_error"        # Incorrect meanings
    ROLE_REVERSAL = "role_reversal"             # Wrong actor/action
    SYSTEM_CONTRADICTION = "system_contradiction" # Wrong system behavior

class QuestionQuality(BaseModel):
    clear_language: bool = Field(..., description="Uses unambiguous language")
    text_based: bool = Field(..., description="Contradiction with text")
    no_tricks: bool = Field(..., description="Clear error to identify")

class GeneratedQuestionAnswerPair(BaseModel):
    """
    Represents a structured QA pair for multi-document comprehension testing.
    """
    # Analysis Fields
    document_extract_analysis: str = Field(
        ...,
        description="Analysis of the key points and structure of the extract",
        min_length=50
    )
    
    chunk_analyses: List[ChunkAnalysis] = Field(
        ...,
        min_items=2,
        description="Analysis of individual chunks and their connections"
    )
    
    testable_concepts: List[str] = Field(
        ...,
        description="Key concepts that can be tested from the extract",
        min_items=2
    )

    potential_question_directions: List[str] = Field(..., description="The possible questions that a human would likely ask")
    best_direction: str = Field(..., description="The best question to ask, a decision made based on the question_directions. Why would it be a good question to ask, and why skills would it test?")

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
        default=QuestionType.FALSE_PREMISE,
        description="Question type (false premise)"
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

1. **Cross-Chunk Contradiction Identification**
   - Identify key facts across chunks
   - Note relationships between chunks
   - Map processes spanning chunks
   - Understand system interactions

2. **Multi-Hop Premise Inversion**
   - Create clear contradictions requiring multiple chunks
   - Ensure text evidence from multiple sources
   - Make errors identifiable through cross-reference
   - Plan corrections requiring synthesis

3. **Complex Question Formation**
   - Present false premise spanning chunks
   - Request correction using multiple sources
   - Enable multi-hop evidence use
   - Support comprehensive explanation

4. **Multi-Source Quality Verification**
   - Check contradiction clarity across chunks
   - Verify text evidence from multiple sources
   - Confirm correction path through chunks
   - Test usefulness of multi-hop reasoning

## Examples

### Example 1: Cross-Chunk Fact Contradiction (Medium)

```json
{
    "document_extract_analysis": "The text discusses climate change impacts across different regions and time periods.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Arctic temperature changes from 1980-2000",
            "relevant_information": "2°C increase in Arctic temperatures",
            "connection_points": ["temperature trends", "timeline connection"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Arctic temperature changes from 2000-2020",
            "relevant_information": "Additional 3°C increase",
            "connection_points": ["temperature trends", "cumulative effects"]
        }
    ],
    "testable_concepts": [
        "temperature changes",
        "cumulative effects",
        "temporal progression"
    ],
    "potential_question_directions": [
        "How do the temperature changes compare across periods?",
        "What is the total temperature increase described?",
        "How do the chunks show progression of change?"
    ],
    "best_direction": "What is the total temperature increase described?",
    "comprehension_type": "fact_contradiction",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "2°C increase in Arctic temperatures from 1980-2000",
        "Additional 3°C increase from 2000-2020"
    ],
    "quote_context": "The quotes establish a total 5°C increase across both periods.",
    "kind": "false-premise",
    "question": "The text indicates that Arctic temperatures increased by only 1°C total from 1980-2020. What's wrong with this premise and what do the texts actually say?",
    "answer": "The texts show a total increase of 5°C: 2°C from 1980-2000 and an additional 3°C from 2000-2020",
    "reasoning": "Requires combining information from both chunks to determine total change.",
    "difficulty": 3,
    "difficulty_justification": "Requires synthesizing numerical information across multiple chunks."
}
```

### Example 2: Multi-Hop Process Inversion (Hard)

```json
{
    "document_extract_analysis": "The text explains photosynthesis and cellular respiration processes.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Photosynthesis process and inputs",
            "relevant_information": "Plants use sunlight to convert CO2 and water into glucose",
            "connection_points": ["glucose production", "energy flow"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Cellular respiration process",
            "relevant_information": "Glucose broken down to release energy",
            "connection_points": ["glucose usage", "energy production"]
        }
    ],
    "testable_concepts": [
        "energy transformation",
        "process sequence",
        "molecular changes"
    ],
    "potential_question_directions": [
        "How do these processes relate to each other?",
        "What is the sequence of energy transformation?",
        "How do materials flow through both processes?"
    ],
    "best_direction": "How do these processes relate to each other?",
    "comprehension_type": "process_inversion",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Plants use sunlight to convert CO2 and water into glucose",
        "Glucose broken down to release energy"
    ],
    "quote_context": "The quotes establish the correct sequence of glucose production then consumption.",
    "kind": "false-premise",
    "question": "The text suggests that cellular respiration produces glucose which is then used in photosynthesis to create energy. What's wrong with this premise and what is the actual sequence described?",
    "answer": "The texts describe the opposite sequence: photosynthesis produces glucose first, which is then broken down during cellular respiration to release energy",
    "reasoning": "Requires understanding the relationship between processes described in separate chunks.",
    "difficulty": 4,
    "difficulty_justification": "Requires synthesizing process information across chunks and understanding causal relationships."
}
```

### Example 3: Cross-Chunk System Contradiction (Very Hard)

```json
{
    "document_extract_analysis": "The text explains ecosystem relationships in a coral reef.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Coral-algae symbiosis",
            "relevant_information": "Corals provide shelter, algae provide nutrients",
            "connection_points": ["symbiotic relationships", "nutrient cycling"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Reef fish interactions",
            "relevant_information": "Fish clean corals and control algae growth",
            "connection_points": ["ecosystem balance", "species interactions"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Environmental impacts",
            "relevant_information": "Temperature affects all relationships",
            "connection_points": ["environmental factors", "system stability"]
        }
    ],
    "testable_concepts": [
        "ecosystem relationships",
        "symbiotic interactions",
        "environmental effects"
    ],
    "potential_question_directions": [
        "How do different species interact in the reef?",
        "What maintains ecosystem balance?",
        "How do environmental factors affect relationships?"
    ],
    "best_direction": "How do different species interact in the reef?",
    "comprehension_type": "system_contradiction",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Corals provide shelter, algae provide nutrients",
        "Fish clean corals and control algae growth",
        "Temperature affects all relationships"
    ],
    "quote_context": "The quotes establish complex interdependencies between species.",
    "kind": "false-premise",
    "question": "The text suggests that coral reef species operate independently, with corals harming algae, fish avoiding corals, and temperature having no effect on these relationships. What's wrong with these premises and how does the system actually work according to the texts?",
    "answer": "The texts describe a highly interdependent system where corals and algae have a mutually beneficial relationship, fish actively help maintain coral health, and temperature affects all these relationships",
    "reasoning": "Requires synthesizing information about multiple relationships across all three chunks.",
    "difficulty": 5,
    "difficulty_justification": "Requires understanding complex system interactions described across multiple chunks and identifying multiple contradictions."
}
```

## Common Pitfalls to Avoid

1. **Single-Chunk Focus**
   ❌ "Contradiction only references one chunk"
   ✅ "Contradiction spans multiple chunks requiring synthesis"

2. **Shallow Connections**
   ❌ "Simple fact checking across chunks"
   ✅ "Deep relationship understanding across chunks"

3. **Isolated Analysis**
   ❌ "Treating chunks independently"
   ✅ "Integrating information across chunks"

4. **Missing Connections**
   ❌ "Failing to link related information"
   ✅ "Explicitly connecting cross-chunk concepts"

## Output Requirements

1. Generate 3-5 multi-hop false premise questions
2. Include questions requiring at least 2 chunks
3. Ensure clear contradictions across chunks
4. Include explicit corrections using multiple sources
5. All premises must be clearly false
6. Questions should test cross-chunk understanding

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Medium/Cross-Chunk Fact)
    },
    {
        // Question 2 (Hard/Multi-Hop Process)
    },
    {
        // Question 3 (Very Hard/System)
    },
    // ...
]
</generated_questions>
```

## Additional Guidelines

1. **Cross-Chunk Premise Selection**
   - Clear contradictions spanning chunks
   - Multi-source errors
   - Synthesis-based corrections
   - Connected concepts

2. **Multi-Hop Error Types**
   - Cross-chunk factual errors
   - Process inversions across chunks
   - Relationship mistakes between chunks
   - System contradictions spanning chunks
   - Definition errors requiring multiple sources

3. **Multi-Source Difficulty Progression**
   - Two-chunk synthesis (Level 2-3)
   - Three-chunk integration (Level 3-4)
   - Complex system understanding (Level 4-5)
   - Multiple relationship errors (Level 5)

4. **Cross-Reference Requirements**
   - Identify errors across chunks
   - Provide multi-source correction
   - Use evidence from multiple chunks
   - Explain cross-chunk contradictions

5. **Multi-Source Evidence Use**
   - Direct quotes from multiple chunks
   - Clear cross-references
   - Explicit corrections using multiple sources
   - Cross-chunk support