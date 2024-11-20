# Multi-Document Conceptual Question Generation

You will receive multiple document extracts from the same source and optionally a summary. Your task is to generate high-quality conceptual questions that probe understanding of core ideas, principles, and relationships presented across the text chunks, requiring synthesis and connection of information from multiple sections.

## Core Principles

1. **Multi-Hop Concept-Based Inquiry**
   - Questions must target key concepts spanning multiple chunks
   - Answers should demonstrate integrated understanding
   - No mere fact recall
   - Focus on cross-chunk principles and relationships

2. **Question Diversity**
   - Cross-chunk principle understanding
   - Multi-section concept application
   - Inter-chunk relationship comprehension
   - Holistic framework understanding
   - Integrated theory explanation
   - Comprehensive model comprehension

3. **Question Quality**
   - Clear conceptual focus across chunks
   - Tests understanding not memory
   - Requires principle application from multiple sections
   - Explores key relationships between chunks

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
    CONCEPTUAL = "conceptual"  # Questions testing understanding of principles and ideas

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Basic concept identification
    EASY = 2         # Simple concept explanation
    MEDIUM = 3       # Concept application
    HARD = 4         # Complex concept relationships
    VERY_HARD = 5    # Deep conceptual synthesis

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
    text_based: bool = Field(..., description="Based on text concepts")
    no_tricks: bool = Field(..., description="Avoids misleading wording")

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
        min_items=2,
        description="Analysis of individual chunks and their connections"
    )
    
    testable_concepts: List[str] = Field(
        ...,
        description="Key concepts that can be tested across extracts",
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
        description="Verbatim quotes from multiple chunks that prove the answer",
        min_items=2
    )
    
    quote_context: str = Field(
        ...,
        description="Explanation of how quotes from different chunks support the answer",
        min_length=30
    )

    # Core Question Fields
    kind: QuestionType = Field(
        default=QuestionType.CONCEPTUAL,
        description="Question type (conceptual)"
    )
    
    question: str = Field(
        ...,
        description="The question requiring multi-chunk synthesis"
    )
    
    answer: str = Field(
        ...,
        description="The correct answer incorporating multiple chunks"
    )
    
    reasoning: str = Field(
        ...,
        description="Detailed explanation of the answer showing cross-chunk integration",
        min_length=50
    )
    
    difficulty: DifficultyLevel = Field(
        ...,
        description="Question difficulty level"
    )
    
    difficulty_justification: str = Field(
        ...,
        description="Explanation of difficulty rating considering multi-chunk synthesis",
        min_length=30
    )

    class Config:
        use_enum_values = True
```

## Question Generation Process

1. **Cross-Chunk Concept Identification**
   - Identify key principles across chunks
   - Map conceptual relationships between sections
   - Note theoretical frameworks spanning chunks
   - Identify core models connecting sections

2. **Multi-Hop Understanding Assessment**
   - Define conceptual focus across chunks
   - Map principle applications between sections
   - Consider relationships spanning chunks
   - Evaluate synthesis depth

3. **Integrated Question Formation**
   - Target key concepts from multiple chunks
   - Require understanding across sections
   - Test relationships between chunks
   - Explore cross-section applications

4. **Multi-Source Quality Verification**
   - Check concept clarity across chunks
   - Verify understanding focus spans sections
   - Confirm text basis from multiple chunks
   - Test synthesis depth

## Examples

### Example 1

```json
{
    "document_extract_analysis": "The text explores quantum computing fundamentals across sections, connecting quantum mechanics principles with computational implementations and algorithmic advantages.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Quantum mechanics foundations",
            "relevant_information": "Explains superposition and entanglement",
            "connection_points": ["Links to qubit behavior", "Connects to quantum gates"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Quantum computing implementation",
            "relevant_information": "Details qubit manipulation and quantum circuits",
            "connection_points": ["References quantum principles", "Links to algorithms"]
        }
    ],
    "testable_concepts": [
        "quantum superposition",
        "entanglement",
        "quantum gates",
        "quantum algorithms",
        "computational advantage"
    ],
    "potential_question_directions": [
        "How does quantum entanglement enable computational advantages?",
        "What role does superposition play in quantum algorithms?",
        "How do quantum gates manipulate qubits for computation?"
    ],
    "best_direction": "How does quantum entanglement enable computational advantages?",
    "comprehension_type": "mechanism_understanding",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Entanglement allows multiple qubits to be correlated in ways impossible for classical bits",
        "Quantum algorithms leverage entanglement to perform parallel computations",
        "The quantum speedup comes from exploiting entangled states"
    ],
    "quote_context": "The quotes demonstrate how entanglement provides the foundation for quantum computational advantages",
    "kind": "conceptual",
    "question": "How does quantum entanglement fundamentally enable quantum computers to achieve computational advantages over classical computers?",
    "answer": "Quantum entanglement allows qubits to be correlated in ways impossible for classical bits, enabling quantum algorithms to perform massive parallel computations through the manipulation of entangled states, leading to exponential speedups for certain problems.",
    "reasoning": "The text explains how entanglement creates quantum correlations that classical computers cannot achieve, and how quantum algorithms specifically exploit these entangled states to perform parallel computations that result in computational speedups.",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding complex quantum mechanical principles and their relationship to computational advantages, synthesizing information from multiple sections."
}
```

### Example 2

```json
{
    "document_extract_analysis": "The text discusses evolutionary biology across sections, linking genetic variation, natural selection mechanisms, and species adaptation patterns.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Genetic variation mechanisms",
            "relevant_information": "Details mutation and recombination",
            "connection_points": ["Links to selection pressure", "Connects to adaptation"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Natural selection processes",
            "relevant_information": "Explains selection mechanisms",
            "connection_points": ["References genetic variation", "Links to fitness"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Species adaptation",
            "relevant_information": "Shows long-term evolutionary changes",
            "connection_points": ["References selection", "Connects to genetics"]
        }
    ],
    "testable_concepts": [
        "genetic variation",
        "natural selection",
        "adaptation mechanisms",
        "evolutionary change",
        "population dynamics"
    ],
    "potential_question_directions": [
        "How do genetic variations influence natural selection?",
        "What role does population size play in evolution?",
        "How do selection pressures drive adaptation?"
    ],
    "best_direction": "How do genetic variations influence natural selection?",
    "comprehension_type": "process_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Genetic mutations provide raw material for evolution",
        "Selection acts on existing variations within populations",
        "Beneficial mutations increase in frequency over generations"
    ],
    "quote_context": "The quotes show the progression from genetic variation to evolutionary change through selection",
    "kind": "conceptual",
    "question": "What is the relationship between genetic variation and the effectiveness of natural selection in driving evolutionary change?",
    "answer": "Genetic variation provides the raw material upon which natural selection acts, with beneficial mutations increasing in frequency over generations, leading to adaptive evolution. Without sufficient genetic variation, natural selection cannot effectively drive evolutionary change.",
    "reasoning": "The text demonstrates how genetic variation creates the foundation for natural selection by providing different traits that can be selected for or against, with beneficial variations becoming more common over time through selective pressures.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding the interplay between genetic mechanisms and selection processes, integrating concepts from multiple sections."
}
```

### Example 3

```json
{
    "document_extract_analysis": "The text examines cognitive development across sections, connecting neuroplasticity, learning mechanisms, and environmental influences.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Brain plasticity fundamentals",
            "relevant_information": "Explains synaptic changes and neural growth",
            "connection_points": ["Links to learning", "Connects to environment"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Learning mechanisms",
            "relevant_information": "Details memory formation and skill acquisition",
            "connection_points": ["References plasticity", "Links to development"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Environmental influences",
            "relevant_information": "Shows impact of experience on brain development",
            "connection_points": ["References learning", "Connects to plasticity"]
        }
    ],
    "testable_concepts": [
        "neuroplasticity",
        "learning mechanisms",
        "environmental factors",
        "cognitive development",
        "synaptic changes"
    ],
    "potential_question_directions": [
        "How does neuroplasticity enable learning?",
        "What role does environment play in brain development?",
        "How do experience and plasticity interact in cognitive development?"
    ],
    "best_direction": "How do experience and plasticity interact in cognitive development?",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Neuroplasticity allows the brain to reorganize based on experience",
        "Environmental stimuli trigger specific patterns of neural activation",
        "Repeated experiences strengthen neural connections through plasticity"
    ],
    "quote_context": "The quotes establish the dynamic interaction between experience and brain plasticity in development",
    "kind": "conceptual",
    "question": "How do environmental experiences and neuroplasticity mechanisms work together to shape cognitive development?",
    "answer": "Environmental experiences trigger specific patterns of neural activation, while neuroplasticity enables the brain to reorganize and strengthen these activated neural connections. This interaction leads to lasting changes in brain structure and function, driving cognitive development.",
    "reasoning": "The text explains how environmental stimuli activate specific neural patterns, and through plasticity mechanisms, these activations lead to structural changes in the brain, creating a dynamic system of experience-dependent development.",
    "difficulty": 5,
    "difficulty_justification": "Requires synthesizing complex interactions between environmental factors, neural mechanisms, and developmental processes, while understanding their dynamic relationships across multiple time scales."
}
```

## Common Pitfalls to Avoid

1. **Single-Chunk Focus**
   ❌ "What is described in section 1?"
   ✅ "How do concepts from sections 1 and 2 interact?"

2. **Shallow Integration**
   ❌ "List the topics from each section."
   ✅ "How do the topics across sections relate to each other?"

3. **Missing Connections**
   ❌ "Explain each section separately."
   ✅ "How do the sections build on each other?"

4. **Oversimplified Synthesis**
   ❌ "What are the main points of all sections?"
   ✅ "How do the main points work together to explain the concept?"

## Output Requirements

1. Generate 3-5 conceptual questions requiring multi-chunk synthesis
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure questions test understanding across chunks
4. Include at least one complex synthesis question
5. All concepts must be from the text chunks
6. Questions should explore relationships between chunks

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Medium/Cross-Chunk Principle)
    },
    {
        // Question 2 (Hard/Multi-Chunk Model)
    },
    {
        // Question 3 (Very Hard/Cross-Section Synthesis)
    },
    // ...
]
</generated_questions>
```

## Additional Guidelines

1. **Cross-Chunk Concept Selection**
   - Choose principles spanning chunks
   - Focus on inter-section relationships
   - Target connecting models
   - Include cross-chunk frameworks

2. **Multi-Hop Understanding Assessment**
   - Test principle application across sections
   - Explore relationships between chunks
   - Examine model use across sections
   - Require multi-chunk synthesis

3. **Difficulty Progression**
   - Basic cross-chunk principles (Level 1-2)
   - Multi-section application (Level 3)
   - Complex inter-chunk relationships (Level 4)
   - Deep cross-section synthesis (Level 5)

4. **Multi-Hop Conceptual Types**
   - Principles: Core ideas spanning chunks
   - Models: Frameworks connecting sections
   - Relationships: Connections between chunks
   - Applications: Using concepts across sections
   - Synthesis: Combining ideas from multiple chunks

5. **Cross-Section Response Evaluation**
   - Clear multi-chunk understanding
   - Cross-section principle application
   - Inter-chunk relationship comprehension
   - Multiple section concept integration