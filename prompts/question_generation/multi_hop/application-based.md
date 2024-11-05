# Multi-Document Application Question Generation

You will receive multiple document extracts from the same source and a summary. Your task is to generate high-quality application questions that test the ability to apply principles, concepts, or processes from multiple text chunks to new situations, requiring multi-hop reasoning.

## Core Principles

1. **Multi-Text Application**
   - Application must use principles from multiple chunks
   - New scenarios must be relatable
   - No external knowledge required
   - Clear connection to content across chunks

2. **Question Diversity**
   - Cross-chunk principle application
   - Process adaptation across sections
   - Integrated model use
   - Multi-step problem solving
   - Combined strategy application
   - Method transfer across contexts

3. **Question Quality**
   - Clear scenario incorporating multiple elements
   - Obvious connections across text chunks
   - Reasonable application
   - Meaningful transfer

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
    APPLICATION = "application"  # Questions requiring application of text concepts

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Simple direct application
    EASY = 2         # Basic principle use
    MEDIUM = 3       # Process adaptation
    HARD = 4         # Complex application
    VERY_HARD = 5    # System-level application

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
    text_based: bool = Field(..., description="Based on text principles")
    no_tricks: bool = Field(..., description="Reasonable application")

class GeneratedQuestionAnswerPair(BaseModel):
    """
    Represents a structured QA pair for multi-document comprehension testing.
    """
    # Analysis Fields
    document_extract_analysis: str = Field(
        ...,
        description="Analysis of the key points and structure across all extracts",
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
        default=QuestionType.APPLICATION,
        description="Question type (application-based)"
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
   - Identify key concepts across chunks
   - Map relationships between chunks
   - Note interconnected processes
   - Understand system interactions

2. **Multi-Hop Scenario Development**
   - Create situations requiring multiple principles
   - Ensure cross-chunk relevance
   - Enable integrated application
   - Maintain coherent connections

3. **Question Formation**
   - Present complex scenario
   - Request multi-step application
   - Guide through connected reasoning
   - Enable comprehensive demonstration

4. **Quality Verification**
   - Check cross-chunk connections
   - Verify integrated applicability
   - Confirm clarity across steps
   - Test comprehensive understanding

## Examples

### Example 1: Cross-Chunk Principle Application (Easy)

```json
{
    "document_extract_analysis": "The text explains both photosynthesis mechanisms and plant growth factors across multiple sections.",
    "chunk_analyses": [
        {
            "chunk_id": "A1",
            "content_summary": "Details of photosynthesis process",
            "relevant_information": "Light energy conversion to chemical energy",
            "connection_points": ["energy production", "carbon dioxide usage"]
        },
        {
            "chunk_id": "A2",
            "content_summary": "Plant growth requirements",
            "relevant_information": "Nutrient uptake and environmental factors",
            "connection_points": ["energy usage", "growth conditions"]
        }
    ],
    "testable_concepts": [
        "energy conversion",
        "growth factors",
        "environmental conditions"
    ],
    "potential_question_directions": [
        "How might varying light conditions affect both photosynthesis and growth in an indoor garden?",
        "What impact would different nutrient levels have on energy production and plant development?",
        "How do temperature changes influence both photosynthesis efficiency and growth rates?"
    ],
    "best_direction": "How might varying light conditions affect both photosynthesis and growth in an indoor garden?",
    "comprehension_type": "principle_application",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Photosynthesis requires light energy for chemical conversion",
        "Plant growth depends on energy availability and environmental conditions"
    ],
    "quote_context": "The quotes connect energy production to growth requirements",
    "kind": "application",
    "question": "Using principles from both passages, how would changing light conditions in an indoor garden affect both photosynthesis and plant growth rates?",
    "answer": "Reduced light would decrease photosynthesis efficiency, leading to less energy production, which would then result in slower growth rates due to limited energy availability",
    "reasoning": "Applies connected principles from both chunks to show cause-effect relationship",
    "difficulty": 2,
    "difficulty_justification": "Straightforward application of related concepts from multiple chunks"
}
```

### Example 2: Multi-Process Adaptation (Medium)

```json
{
    "document_extract_analysis": "The text covers both cellular respiration and energy distribution systems in organisms.",
    "chunk_analyses": [
        {
            "chunk_id": "B1",
            "content_summary": "Cellular respiration process",
            "relevant_information": "Energy extraction from glucose",
            "connection_points": ["ATP production", "oxygen use"]
        },
        {
            "chunk_id": "B2",
            "content_summary": "Energy distribution in organisms",
            "relevant_information": "ATP transport and usage",
            "connection_points": ["energy distribution", "cellular needs"]
        }
    ],
    "testable_concepts": [
        "energy extraction",
        "distribution systems",
        "cellular efficiency"
    ],
    "potential_question_directions": [
        "How would altitude changes affect both energy production and distribution in athletes?",
        "What adaptations might occur in both systems during extended exercise?",
        "How do temperature changes impact both energy generation and distribution?"
    ],
    "best_direction": "How would altitude changes affect both energy production and distribution in athletes?",
    "comprehension_type": "process_adaptation",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Cellular respiration requires oxygen for maximum efficiency",
        "Energy distribution depends on ATP availability and transport"
    ],
    "quote_context": "The quotes establish the connection between oxygen availability and energy systems",
    "kind": "application",
    "question": "Using the principles from both passages, how would high-altitude training affect an athlete's cellular energy production and distribution systems?",
    "answer": "Lower oxygen availability would reduce cellular respiration efficiency, leading to decreased ATP production, which would then impact the energy distribution system's ability to supply working muscles",
    "reasoning": "Integrates understanding of both energy production and distribution processes in a new context",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding and connecting multiple biological processes in a specific scenario"
}
```

### Example 3: System-Level Integration (Very Hard)

```json
{
    "document_extract_analysis": "The text explains both ecosystem nutrient cycles and climate impact on biodiversity.",
    "chunk_analyses": [
        {
            "chunk_id": "C1",
            "content_summary": "Nutrient cycling in ecosystems",
            "relevant_information": "Carbon and nitrogen cycles",
            "connection_points": ["nutrient flow", "biological processes"]
        },
        {
            "chunk_id": "C2",
            "content_summary": "Climate effects on biodiversity",
            "relevant_information": "Species adaptation and interaction",
            "connection_points": ["environmental change", "species response"]
        }
    ],
    "testable_concepts": [
        "nutrient cycles",
        "climate impacts",
        "ecosystem adaptation"
    ],
    "potential_question_directions": [
        "How might warming temperatures affect both nutrient cycling and species interactions in a forest ecosystem?",
        "What cascading effects would altered precipitation have on both nutrient availability and biodiversity?",
        "How do seasonal changes impact both nutrient flows and species adaptation?"
    ],
    "best_direction": "How might warming temperatures affect both nutrient cycling and species interactions in a forest ecosystem?",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Temperature influences rate of nutrient cycling",
        "Species interactions depend on resource availability",
        "Climate changes affect both chemical and biological processes"
    ],
    "quote_context": "The quotes link environmental changes to both chemical and biological systems",
    "kind": "application",
    "question": "Using principles from both passages, analyze how a 2°C temperature increase would affect both nutrient cycling and species interactions in a forest ecosystem over a decade.",
    "answer": "Warmer temperatures would accelerate nutrient cycling, potentially leading to initial increased availability but possible long-term depletion, while simultaneously affecting species interactions through changed resource availability and timing mismatches",
    "reasoning": "Requires integration of multiple system-level processes and their interactions over time",
    "difficulty": 5,
    "difficulty_justification": "Demands understanding of complex system interactions and long-term effects across multiple domains"
}
```

## Common Pitfalls to Avoid

1. **Isolated Analysis**
   ❌ "Focus on single chunk principles"
   ✅ "Integrate principles across chunks"

2. **Superficial Connections**
   ❌ "Loose associations between concepts"
   ✅ "Meaningful integration of related principles"

3. **Overcomplicated Scenarios**
   ❌ "Too many variables across chunks"
   ✅ "Clear, focused multi-principle application"

4. **Disconnected Applications**
   ❌ "Separate applications for each chunk"
   ✅ "Integrated application across chunks"

## Output Requirements

1. Generate 3-5 multi-hop application questions
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure clear connections across chunks
4. Include realistic scenarios requiring multiple principles
5. Provide valid integrated application paths
6. Scale difficulty appropriately

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Easy/Cross-Principle)
    },
    {
        // Question 2 (Medium/Multi-Process)
    },
    {
        // Question 3 (Hard/System-Integration)
    },
    // ...
]
</generated_questions>
```

## Additional Guidelines

1. **Cross-Chunk Integration**
   - Identify related principles
   - Map concept connections
   - Ensure coherent integration
   - Maintain text fidelity

2. **Multi-Hop Scenario Design**
   - Create integrated situations
   - Scale complexity appropriately
   - Ensure multi-principle relevance
   - Enable comprehensive demonstration

3. **Difficulty Progression**
   - Simple cross-principle (Level 1-2)
   - Multi-process integration (Level 3)
   - Complex system application (Level 4)
   - Full system integration (Level 5)

4. **Application Types**
   - Cross-chunk principles
   - Integrated processes
   - Connected models
   - Multi-step methods
   - System-level theories

5. **Response Evaluation**
   - Clear integration path
   - Multi-principle adherence
   - Reasonable outcomes
   - Valid demonstration across chunks