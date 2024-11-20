# Multi-Document Edge Case Question Generation

You will receive multiple document extracts from the same source document and a summary. Your task is to generate high-quality edge case questions that test understanding of boundary conditions, exceptions, and special cases that require synthesizing information across multiple chunks of text.

## Core Principles

1. **Cross-Text Boundaries**
   - Edge cases must span multiple chunks
   - Exceptions must be synthesized
   - Special conditions connected
   - Boundary conditions integrated

2. **Question Diversity**
   - Multi-hop exception cases
   - Cross-chunk boundaries
   - Synthesized circumstances
   - Connected rule limitations
   - Compound extreme cases
   - Inter-related conditions

3. **Question Quality**
   - Clear edge identification across chunks
   - Multi-text supported exceptions
   - Valid synthesized cases
   - Connected boundaries

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
    EDGE_CASE = "edge-case"  # Questions exploring boundaries and exceptions

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Simple exception
    EASY = 2         # Basic boundary case
    MEDIUM = 3       # Complex exception
    HARD = 4         # Multiple conditions
    VERY_HARD = 5    # System edge cases

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
    text_based: bool = Field(..., description="Edge case from text")
    no_tricks: bool = Field(..., description="Valid boundary case")

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
        description="Key concepts that can be tested across chunks",
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
        description="Verbatim quotes from multiple chunks that prove the answer",
        min_items=1
    )
    
    quote_context: str = Field(
        ...,
        description="Explanation of how quotes from different chunks support the answer",
        min_length=30
    )

    # Core Question Fields
    kind: QuestionType = Field(
        default=QuestionType.EDGE_CASE,
        description="Question type (edge-case)"
    )
    
    question: str = Field(
        ...,
        description="The multi-hop question"
    )
    
    answer: str = Field(
        ...,
        description="The correct answer synthesizing information across chunks"
    )
    
    reasoning: str = Field(
        ...,
        description="Detailed explanation of the answer showing connections",
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

1. **Cross-Chunk Edge Identification**
   - Spot connected exceptions
   - Find related boundaries
   - Note linked limitations
   - Identify synthesized cases

2. **Multi-Hop Case Development**
   - Define conditions across chunks
   - Map cross-text implications
   - Consider inter-chunk interactions
   - Verify multi-text support

3. **Synthesized Question Formation**
   - Present connected edges
   - Request integrated understanding
   - Enable cross-chunk explanation
   - Guide multi-hop analysis

4. **Quality Verification**
   - Check multi-text basis
   - Verify cross-chunk validity
   - Confirm synthesis clarity
   - Test connection usefulness

## Examples

### Example 1

```json
{
    "document_extract_analysis": "The texts explore quantum computing principles, focusing on qubit behavior under different conditions and error correction mechanisms.",
    "chunk_analyses": [
        {
            "chunk_id": "Q1",
            "content_summary": "Qubit decoherence patterns",
            "relevant_information": "Qubits maintain coherence for microseconds",
            "connection_points": ["quantum states", "time limitations"]
        },
        {
            "chunk_id": "Q2",
            "content_summary": "Error correction methods",
            "relevant_information": "Surface codes require physical qubit overhead",
            "connection_points": ["error rates", "resource requirements"]
        }
    ],
    "testable_concepts": [
        "quantum coherence",
        "error correction thresholds",
        "resource scaling"
    ],
    "potential_question_directions": [
        "How do decoherence times affect error correction requirements?",
        "What trade-offs exist between physical qubits and error rates?",
        "When do error correction methods become impractical?"
    ],
    "best_direction": "How do decoherence times affect error correction requirements?",
    "comprehension_type": "process_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Qubits lose coherence within microseconds",
        "Surface code correction requires 1000 physical qubits per logical qubit",
        "Error rates must stay below 1% for effective correction"
    ],
    "quote_context": "The quotes establish the relationship between coherence time, error rates, and resource requirements",
    "kind": "edge-case",
    "question": "How does the microsecond-scale coherence time of qubits influence the minimum number of physical qubits needed for effective error correction?",
    "answer": "The brief coherence time leads to higher error rates, requiring at least 1000 physical qubits per logical qubit to implement surface code correction effectively when error rates approach 1%",
    "reasoning": "The question tests understanding of how temporal limitations create resource requirements through their effect on error rates",
    "difficulty": 1,
    "difficulty_justification": "Requires simple connection between time limitations and resource needs"
}
```

### Example 2

```json
{
    "document_extract_analysis": "The passages detail neural network training dynamics, focusing on gradient descent behavior in different optimization scenarios.",
    "chunk_analyses": [
        {
            "chunk_id": "N1",
            "content_summary": "Learning rate effects",
            "relevant_information": "High rates cause divergence",
            "connection_points": ["optimization stability", "convergence speed"]
        },
        {
            "chunk_id": "N2",
            "content_summary": "Batch size impact",
            "relevant_information": "Larger batches affect gradient noise",
            "connection_points": ["training stability", "computational efficiency"]
        },
        {
            "chunk_id": "N3",
            "content_summary": "Memory constraints",
            "relevant_information": "GPU memory limits batch size",
            "connection_points": ["hardware limitations", "training options"]
        }
    ],
    "testable_concepts": [
        "optimization dynamics",
        "resource constraints",
        "training stability"
    ],
    "potential_question_directions": [
        "How do hardware constraints affect training strategies?",
        "What relationships exist between batch size and learning rate?",
        "When do multiple constraints force training compromises?"
    ],
    "best_direction": "How do hardware constraints affect training strategies?",
    "comprehension_type": "condition_interaction",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Learning rates above 0.1 often cause divergence",
        "Larger batches reduce gradient noise but increase memory usage",
        "GPU memory limits maximum batch size to 256 samples"
    ],
    "quote_context": "The quotes establish interactions between hardware constraints, training parameters, and optimization behavior",
    "kind": "edge-case",
    "question": "When GPU memory limits batch size to 256 samples, what adjustments to the learning rate become necessary to maintain training stability, and why?",
    "answer": "With batch size limited to 256, learning rates must be reduced below 0.1 to compensate for increased gradient noise, maintaining stability despite suboptimal conditions",
    "reasoning": "Tests understanding of how hardware constraints create cascading effects on training parameters and stability",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding complex interactions between hardware limitations and training dynamics"
}
```

### Example 3

```json
{
    "document_extract_analysis": "The texts examine cellular stress responses, focusing on heat shock protein expression and metabolic adaptations.",
    "chunk_analyses": [
        {
            "chunk_id": "C1",
            "content_summary": "Heat shock response",
            "relevant_information": "HSP70 expression timing",
            "connection_points": ["protein protection", "stress threshold"]
        },
        {
            "chunk_id": "C2",
            "content_summary": "Metabolic changes",
            "relevant_information": "ATP consumption patterns",
            "connection_points": ["energy usage", "cellular resources"]
        },
        {
            "chunk_id": "C3",
            "content_summary": "Recovery phases",
            "relevant_information": "Protein refolding requirements",
            "connection_points": ["damage repair", "resource allocation"]
        }
    ],
    "testable_concepts": [
        "stress response timing",
        "resource allocation",
        "recovery mechanisms"
    ],
    "potential_question_directions": [
        "How do cells prioritize responses under limited resources?",
        "What trade-offs occur during stress recovery?",
        "When do multiple stressors overwhelm cellular responses?"
    ],
    "best_direction": "How do cells prioritize responses under limited resources?",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "HSP70 expression peaks within 30 minutes",
        "ATP reserves deplete by 60% during stress",
        "Protein refolding requires sustained ATP availability"
    ],
    "quote_context": "The quotes establish temporal and resource constraints on cellular stress responses",
    "kind": "edge-case",
    "question": "If ATP reserves deplete by 60% during the first 30 minutes of stress response, how does this affect the cell's ability to maintain HSP70 expression and protein refolding activities simultaneously?",
    "answer": "The severe ATP depletion forces cells to prioritize immediate HSP70 expression over protein refolding, creating a backlog of damaged proteins that extends recovery time",
    "reasoning": "Tests understanding of how resource limitations create competing demands between immediate stress response and recovery processes",
    "difficulty": 4,
    "difficulty_justification": "Requires analyzing complex resource allocation decisions and their consequences across multiple cellular processes"
}
```

## Common Pitfalls to Avoid

1. **Single-Chunk Focus**
   ❌ "What exception appears in the first passage?"
   ✅ "How do the exceptions described across passages interact?"

2. **Disconnected Analysis**
   ❌ "What are the separate conditions in each text?"
   ✅ "How do the conditions connect across texts?"

3. **Missing Synthesis**
   ❌ "List the boundary conditions from each passage"
   ✅ "How do the boundary conditions interact across passages?"

4. **Isolated Consideration**
   ❌ "What happens in this specific case?"
   ✅ "How do multiple cases combine to create new effects?"

## Output Requirements

1. Generate 3-5 multi-hop edge case questions
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure cases span multiple chunks
4. Include clear cross-chunk connections
5. Provide specific synthesis criteria
6. Scale complexity appropriately

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Easy/Cross-Chunk Exception)
    },
    {
        // Question 2 (Medium/Multi-Hop Boundary)
    },
    {
        // Question 3 (Hard/System Synthesis)
    },
    // ...
]
</generated_questions>
```

## Additional Guidelines

1. **Cross-Chunk Edge Selection**
   - Find connected exceptions
   - Identify related boundaries
   - Note interacting conditions
   - Map system connections

2. **Multi-Hop Development**
   - Define cross-chunk conditions
   - Support with multiple texts
   - Explain interactions
   - Show emergent effects

3. **Difficulty Progression**
   - Simple connections (Level 1-2)
   - Complex interactions (Level 3)
   - Multiple syntheses (Level 4)
   - System interactions (Level 5)

4. **Multi-Hop Edge Types**
   - Connected exceptions
   - Interacting boundaries
   - Compound circumstances
   - System interactions
   - Emergent conditions
   - Synthetic cases

5. **Response Evaluation**
   - Clear connections
   - Multi-text support
   - Valid synthesis
   - Proper interaction