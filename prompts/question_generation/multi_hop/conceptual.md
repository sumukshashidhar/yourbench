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

### Example 1: Cross-Chunk Principle Understanding (Medium)

```json
{
    "document_extract_analysis": "The text explains climate change across multiple sections, connecting atmospheric composition changes with global temperature effects and ecosystem impacts.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Explains greenhouse gas composition changes",
            "relevant_information": "Details CO2 increase and atmospheric changes",
            "connection_points": ["Links to temperature effects", "Connects to industrial activities"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Describes global temperature impacts",
            "relevant_information": "Shows temperature rise patterns and effects",
            "connection_points": ["References gas composition", "Links to ecosystem changes"]
        }
    ],
    "testable_concepts": [
        "greenhouse effect",
        "atmospheric composition",
        "temperature change",
        "climate impact"
    ],
    "potential_question_directions": [
        "How do changes in atmospheric composition affect global temperatures?",
        "What is the relationship between industrial activities and climate change?",
        "How do greenhouse gases influence global temperature patterns?"
    ],
    "best_direction": "How do changes in atmospheric composition affect global temperatures?",
    "comprehension_type": "relationship_comprehension",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Greenhouse gas concentrations have increased by 40% since pre-industrial times",
        "Global temperatures have risen by 1°C corresponding to atmospheric changes",
        "The relationship between CO2 and temperature shows a clear correlation"
    ],
    "quote_context": "The quotes establish the connection between atmospheric changes and temperature effects across multiple sections",
    "kind": "conceptual",
    "question": "Based on the text's explanation, how do industrial-era changes in atmospheric composition contribute to global temperature increases?",
    "answer": "Industrial activities have increased greenhouse gas concentrations by 40%, which trap more heat in the atmosphere, leading to a measured 1°C rise in global temperatures",
    "reasoning": "The text connects industrial activities to increased greenhouse gas concentrations, which then affect global temperatures through the greenhouse effect mechanism",
    "difficulty": 3,
    "difficulty_justification": "Requires synthesizing information about atmospheric composition and temperature effects from multiple sections"
}
```

### Example 2: Multi-Chunk Model Comprehension (Hard)

```json
{
    "document_extract_analysis": "The text describes neural network learning across multiple sections, connecting initial network structure with training processes and final behavior.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Explains neural network architecture",
            "relevant_information": "Details network layers and connections",
            "connection_points": ["Links to training process", "Connects to data flow"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Describes training methodology",
            "relevant_information": "Explains backpropagation and weight updates",
            "connection_points": ["References network structure", "Links to learning outcomes"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Outlines network behavior",
            "relevant_information": "Shows how trained networks process new data",
            "connection_points": ["References training process", "Connects to architecture"]
        }
    ],
    "testable_concepts": [
        "neural architecture",
        "network training",
        "learning process",
        "data processing"
    ],
    "potential_question_directions": [
        "How does network architecture influence the training process?",
        "What role does backpropagation play in network learning?",
        "How do initial network structure and training combine to determine final behavior?"
    ],
    "best_direction": "How do initial network structure and training combine to determine final behavior?",
    "comprehension_type": "model_comprehension",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Network architecture defines possible connection patterns",
        "Training adjusts connection weights through backpropagation",
        "Final behavior emerges from the combination of structure and learned weights"
    ],
    "quote_context": "The quotes show how network structure and training process interact to produce final behavior",
    "kind": "conceptual",
    "question": "According to the text, how do the initial network architecture and training process work together to create the final behavioral capabilities of a neural network?",
    "answer": "The initial architecture defines possible connection patterns, while training adjusts these connections' weights through backpropagation, together determining how the network processes new information",
    "reasoning": "The text explains how network structure provides the framework, while training optimizes this structure through weight adjustments, jointly creating the network's processing capabilities",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding and integrating complex relationships between network structure, training, and behavior from multiple sections"
}
```

### Example 3: Cross-Section Concept Synthesis (Very Hard)

```json
{
    "document_extract_analysis": "The text explains ecosystem resilience across multiple sections, connecting species diversity, environmental pressures, and adaptation mechanisms.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Explains biodiversity importance",
            "relevant_information": "Details species interactions and roles",
            "connection_points": ["Links to environmental stress", "Connects to adaptation"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Describes environmental pressures",
            "relevant_information": "Shows how changes affect ecosystems",
            "connection_points": ["References species diversity", "Links to resilience"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Outlines adaptation mechanisms",
            "relevant_information": "Explains how ecosystems adapt over time",
            "connection_points": ["References environmental pressure", "Connects to diversity"]
        }
    ],
    "testable_concepts": [
        "biodiversity",
        "environmental pressure",
        "ecosystem adaptation",
        "system resilience"
    ],
    "potential_question_directions": [
        "How does biodiversity contribute to ecosystem resilience?",
        "What role do adaptation mechanisms play in response to environmental pressure?",
        "How do species diversity and adaptation mechanisms together determine ecosystem stability?"
    ],
    "best_direction": "How do species diversity and adaptation mechanisms together determine ecosystem stability?",
    "comprehension_type": "concept_synthesis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Species diversity provides multiple response pathways to stress",
        "Environmental pressures trigger various adaptation mechanisms",
        "Ecosystem stability depends on both diversity and adaptation capacity"
    ],
    "quote_context": "The quotes establish how diversity and adaptation work together for ecosystem stability",
    "kind": "conceptual",
    "question": "Based on the text's explanation, how do species diversity and adaptation mechanisms interact to maintain ecosystem stability under environmental pressure?",
    "answer": "Species diversity provides multiple possible response pathways, while adaptation mechanisms enable species to adjust to changes, together creating a resilient system that can maintain stability under various environmental pressures",
    "reasoning": "The text shows how diversity creates response options and adaptation mechanisms enable actual changes, working together to maintain ecosystem function under stress",
    "difficulty": 5,
    "difficulty_justification": "Requires synthesizing complex relationships between diversity, adaptation, and stability from multiple sections while understanding their dynamic interaction"
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