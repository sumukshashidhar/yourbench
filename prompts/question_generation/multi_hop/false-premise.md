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

### Example 1
```json
{
    "document_extract_analysis": "The text explores the evolution of early computing systems, focusing on ENIAC, UNIVAC, and their impact on modern computing architecture.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "ENIAC development and specifications",
            "relevant_information": "ENIAC was completed in 1945, used vacuum tubes, required manual rewiring",
            "connection_points": ["technological evolution", "programming methods"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "UNIVAC innovations",
            "relevant_information": "UNIVAC introduced stored programs in 1951, eliminated manual rewiring",
            "connection_points": ["programming evolution", "architectural changes"]
        }
    ],
    "testable_concepts": [
        "computer architecture evolution",
        "programming paradigm shifts",
        "technological transitions",
        "historical progression"
    ],
    "potential_question_directions": [
        "How did programming methods evolve between systems?",
        "What were the key architectural differences?",
        "How did storage mechanisms change?",
        "What impact did these changes have on computing?"
    ],
    "best_direction": "How did programming methods evolve between systems?",
    "comprehension_type": "process_evolution",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "ENIAC required manual rewiring for each new program",
        "UNIVAC introduced stored programs, eliminating manual rewiring"
    ],
    "quote_context": "The quotes demonstrate the fundamental shift from physical to stored program computing",
    "kind": "false-premise",
    "question": "How accurate is the claim that ENIAC introduced stored program computing while UNIVAC still required manual rewiring for new programs?",
    "answer": "This claim inverts the historical reality. ENIAC required manual rewiring for new programs, while UNIVAC actually introduced stored program computing in 1951, marking a significant advance in computer architecture.",
    "reasoning": "The question requires understanding the chronological development of programming methods and recognizing that UNIVAC's introduction of stored programs was a progression from ENIAC's manual rewiring approach.",
    "difficulty": 3,
    "difficulty_justification": "Requires synthesizing historical information and understanding technological progression across multiple chunks while identifying reversed attribution of innovations."
}
```

### Example 2
```json
{
    "document_extract_analysis": "The text details the nitrogen cycle in ecosystems, focusing on bacterial roles and plant interactions.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Nitrogen fixation process",
            "relevant_information": "Bacteria convert atmospheric N2 to usable forms",
            "connection_points": ["bacterial processes", "nitrogen transformation"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Plant nitrogen utilization",
            "relevant_information": "Plants absorb fixed nitrogen from soil",
            "connection_points": ["nutrient uptake", "plant growth"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Decomposition cycle",
            "relevant_information": "Dead organisms return nitrogen to soil",
            "connection_points": ["nutrient recycling", "ecosystem balance"]
        }
    ],
    "testable_concepts": [
        "nitrogen transformation",
        "ecosystem cycles",
        "bacterial-plant relationships",
        "nutrient flow patterns"
    ],
    "potential_question_directions": [
        "How does nitrogen move through the ecosystem?",
        "What roles do different organisms play?",
        "How are processes interconnected?",
        "What maintains cycle balance?"
    ],
    "best_direction": "How does nitrogen move through the ecosystem?",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Bacteria convert atmospheric N2 to usable forms",
        "Plants absorb fixed nitrogen from soil",
        "Dead organisms return nitrogen to soil"
    ],
    "quote_context": "The quotes establish the complete nitrogen cycle pathway through multiple organisms",
    "kind": "false-premise",
    "question": "The text suggests that plants directly convert atmospheric nitrogen to usable forms, while bacteria only decompose dead matter. What's incorrect about this understanding?",
    "answer": "This completely misrepresents the process. According to the text, bacteria (not plants) convert atmospheric nitrogen to usable forms, plants then absorb this fixed nitrogen from the soil, and decomposition returns nitrogen to the soil.",
    "reasoning": "Understanding requires synthesizing information about multiple organisms' roles and their sequence in the nitrogen cycle across all three chunks.",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding complex system interactions, process sequences, and correcting multiple role reversals across three text chunks."
}
```

### Example 3
```json
{
    "document_extract_analysis": "The text examines quantum entanglement and its implications for quantum computing.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Quantum entanglement definition",
            "relevant_information": "Particles remain connected regardless of distance",
            "connection_points": ["quantum properties", "particle relationships"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Quantum computing applications",
            "relevant_information": "Entanglement enables parallel processing",
            "connection_points": ["practical applications", "computational advantages"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Measurement effects",
            "relevant_information": "Measuring one particle affects its entangled partner",
            "connection_points": ["observation impacts", "quantum behavior"]
        }
    ],
    "testable_concepts": [
        "quantum entanglement",
        "measurement effects",
        "computational implications",
        "particle behavior"
    ],
    "potential_question_directions": [
        "How does entanglement affect computation?",
        "What happens during measurement?",
        "How do particles maintain connections?",
        "What enables quantum advantages?"
    ],
    "best_direction": "How does entanglement affect computation?",
    "comprehension_type": "mechanism_understanding",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Particles remain connected regardless of distance",
        "Entanglement enables parallel processing",
        "Measuring one particle affects its entangled partner"
    ],
    "quote_context": "The quotes establish the fundamental properties and computational implications of entanglement",
    "kind": "false-premise",
    "question": "In what ways does the text's portrayal of quantum entanglement as a hindrance to computational speed, due to the supposed necessity of physical connectivity between particles, misrepresent the actual advantages provided by entanglement in quantum computing?",
    "answer": "The portrayal is incorrect as quantum entanglement facilitates faster parallel processing, does not require physical connectivity between particles, and measuring one particle indeed affects its entangled partner, contrary to the text's claims.",
    "reasoning": "This question demands a deep understanding of quantum entanglement's role in enhancing computational speed and efficiency, challenging the misconception of physical connectivity and measurement effects, as explained across multiple text chunks.",
    "difficulty": 5,
    "difficulty_justification": "Requires synthesizing complex physics concepts across three chunks while identifying multiple fundamental misconceptions about quantum behavior."
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