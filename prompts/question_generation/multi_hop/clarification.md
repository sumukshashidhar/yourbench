# Multi-Document Clarification Question Generation

You will receive multiple document extracts from the same document and a summary. Your task is to generate high-quality clarification questions that identify areas needing further explanation or understanding across the provided text extracts, focusing on connections and relationships between different chunks.

## Core Principles

1. **Multi-Hop Clarity Focus**
   - Questions target potentially unclear elements across chunks
   - Answers must exist within multiple text chunks
   - Focuses on understanding connections and relationships
   - Identifies cross-chunk ambiguity or complexity

2. **Question Diversity**
   - Term clarification across contexts
   - Process explanation spanning chunks
   - Relationship clarification between sections
   - Context questions linking information
   - Detail explanation across passages
   - Reference resolution between chunks

3. **Question Quality**
   - Multiple focus points across chunks
   - Clear need for cross-chunk explanation
   - Answerable from combined texts
   - Meaningful multi-hop clarification

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
    CLARIFICATION = "clarification"  # Questions seeking deeper understanding of specific elements

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Basic term clarification
    EASY = 2         # Simple explanation needs
    MEDIUM = 3       # Process understanding
    HARD = 4         # Complex relationship clarity
    VERY_HARD = 5    # System-level understanding

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
    text_based: bool = Field(..., description="Clarification exists in text")
    no_tricks: bool = Field(..., description="Genuine clarity need")

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
        default=QuestionType.CLARIFICATION,
        description="Question type (clarification)"
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

1. **Cross-Chunk Clarity Need Identification**
   - Spot complex terms across chunks
   - Note unclear processes spanning sections
   - Find ambiguous references between chunks
   - Identify assumed knowledge connections

2. **Multi-Hop Clarification Mapping**
   - Locate explanations across chunks
   - Find context clues between sections
   - Track references across passages
   - Check completeness of connected information

3. **Cross-Reference Question Formation**
   - Target specific elements across chunks
   - Frame for clarity in connections
   - Enable precise multi-hop answers
   - Support understanding of relationships

4. **Multi-Hop Quality Verification**
   - Check answer presence across chunks
   - Verify clarity need in connections
   - Confirm cross-reference specificity
   - Test usefulness of relationship understanding

## Examples

### Example 1

```json
{
    "document_extract_analysis": "The text explores quantum entanglement across multiple sections, covering theoretical foundations and practical applications.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Introduces quantum entanglement basics",
            "relevant_information": "Definition and properties of entangled particles",
            "connection_points": ["Measurement effects", "Particle relationships"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Details measurement impact",
            "relevant_information": "How measuring one particle affects another",
            "connection_points": ["Theoretical implications", "Practical applications"]
        }
    ],
    "testable_concepts": [
        "quantum entanglement principles",
        "measurement effects",
        "particle relationships",
        "quantum mechanics fundamentals"
    ],
    "potential_question_directions": [
        "How does measurement of one particle influence its entangled partner?",
        "What makes quantum entanglement different from classical particle behavior?",
        "Why is entanglement considered 'spooky action at a distance'?"
    ],
    "best_direction": "How does measurement of one particle influence its entangled partner?",
    "comprehension_type": "mechanism_understanding",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Entangled particles share a quantum state regardless of distance",
        "Measuring one particle instantly affects its partner's state",
        "This phenomenon defies classical physics principles"
    ],
    "quote_context": "The quotes establish the unique relationship between entangled particles and measurement effects",
    "kind": "clarification",
    "question": "What happens to an entangled particle pair when one particle is measured, and why is this behavior significant?",
    "answer": "When one entangled particle is measured, its partner's state is instantly determined, regardless of distance. This instantaneous influence defies classical physics and demonstrates quantum mechanics' unique properties.",
    "reasoning": "The answer combines fundamental principles from the first chunk with measurement effects from the second to explain this counter-intuitive phenomenon",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding abstract quantum concepts and connecting theoretical principles with practical effects"
}
```

### Example 2

```json
{
    "document_extract_analysis": "The text examines climate feedback loops across multiple sections, focusing on interconnected environmental systems.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Describes Arctic ice melt feedback",
            "relevant_information": "Ice reflectivity and heat absorption",
            "connection_points": ["Temperature effects", "Ocean circulation"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Details ocean circulation changes",
            "relevant_information": "Current patterns and heat distribution",
            "connection_points": ["Global impacts", "System interactions"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Explains atmospheric consequences",
            "relevant_information": "Weather pattern shifts",
            "connection_points": ["Climate stability", "Global effects"]
        }
    ],
    "testable_concepts": [
        "feedback mechanisms",
        "system interactions",
        "environmental change",
        "global impacts"
    ],
    "potential_question_directions": [
        "In what ways do Arctic ice changes catalyze a series of environmental responses that extend beyond immediate geographical boundaries?",
        "What connects ocean circulation to atmospheric patterns?",
        "Why do small changes amplify through the climate system?"
    ],
    "best_direction": "In what ways do Arctic ice changes catalyze a series of environmental responses that extend beyond immediate geographical boundaries?",
    "comprehension_type": "chain_effect",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Melting ice reduces surface reflectivity, increasing heat absorption",
        "Changed temperature gradients affect ocean current patterns",
        "Altered circulation influences global weather systems"
    ],
    "quote_context": "The quotes trace the cascade of effects from ice melt through ocean and atmospheric systems",
    "kind": "clarification",
    "question": "In what ways do Arctic ice changes catalyze a series of environmental responses that extend beyond immediate geographical boundaries, and how do these changes impact global climate patterns?",
    "answer": "Arctic ice melt reduces surface reflectivity, increasing heat absorption. This affects ocean temperature gradients, altering circulation patterns, which in turn influences global weather systems and climate stability.",
    "reasoning": "The answer integrates information across all chunks to explain the interconnected chain of environmental responses",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding complex system interactions and tracing multiple cause-effect relationships across different environmental components"
}
```

### Example 3

```json
{
    "document_extract_analysis": "The text discusses neural network architecture and learning processes across multiple sections.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Describes network structure",
            "relevant_information": "Layer organization and connections",
            "connection_points": ["Information flow", "Weight adjustments"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Explains backpropagation",
            "relevant_information": "Error calculation and weight updates",
            "connection_points": ["Learning process", "Optimization"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Details optimization techniques",
            "relevant_information": "Learning rate and momentum",
            "connection_points": ["Training efficiency", "Error reduction"]
        }
    ],
    "testable_concepts": [
        "neural architecture",
        "learning algorithms",
        "optimization methods",
        "network training"
    ],
    "potential_question_directions": [
        "Why might certain network architectures fail despite optimal backpropagation?",
        "What role do optimization techniques play in training efficiency?",
        "How does information flow through the network during training?"
    ],
    "best_direction": "Why might certain network architectures fail despite optimal backpropagation?",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Network layers process information forward through weighted connections",
        "Backpropagation calculates errors and updates weights backward through layers",
        "Optimization techniques control weight adjustment magnitude and direction"
    ],
    "quote_context": "The quotes establish the critical interdependence between architecture and learning mechanisms",
    "kind": "clarification",
    "question": "Why might a neural network with theoretically sound backpropagation and optimization techniques still fail to learn effectively, and what architectural considerations could be responsible for this breakdown in the learning process?",
    "answer": "A neural network might fail despite proper backpropagation and optimization when its architecture creates inherent limitations, such as vanishing gradients in deep networks or insufficient layer capacity. The network's structure determines how effectively error signals can propagate backward, and even optimal weight adjustments cannot overcome fundamental architectural constraints on information flow.",
    "reasoning": "The answer integrates architectural limitations with learning mechanisms to explain why structural decisions can override even well-implemented training processes, highlighting the complex interplay between network design and learning capability",
    "difficulty": 5,
    "difficulty_justification": "Requires deep understanding of neural network failure modes and the complex relationships between architecture, learning algorithms, and optimization techniques"
}
```

## Common Pitfalls to Avoid

1. **Single-Chunk Focus**
   ❌ "What are B-cells?"
   ✅ "How do B-cells and T-cells coordinate their responses?"

2. **Missing Connections**
   ❌ "What do T-cells do?"
   ✅ "How do T-cell signals influence B-cell and memory cell development?"

3. **Shallow Integration**
   ❌ "List all immune cells mentioned"
   ✅ "Explain how different immune cells work together through signaling"

4. **Ignoring Relationships**
   ❌ "What is cytokine signaling?"
   ✅ "How does cytokine signaling enable cell-to-cell coordination?"

## Output Requirements

1. Generate 3-5 cross-chunk clarification questions
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure questions target specific clarity needs across chunks
4. Include clear text-based answers drawing from multiple chunks
5. Questions should help resolve cross-chunk ambiguity
6. Balance difficulty levels

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Easy/Term)
    },
    {
        // Question 2 (Medium/Process)
    },
    {
        // Question 3 (Hard/System)
    },
    // ...
]
</generated_questions>
```

## Additional Guidelines

1. **Cross-Chunk Clarity Need Selection**
   - Technical terms across contexts
   - Complex processes spanning chunks
   - Relationships between sections
   - System interactions across passages
   - Cross-references
   - Connected context needs

2. **Multi-Hop Question Formation**
   - Target multiple chunks
   - Focus on connections
   - Enable integrated answers
   - Support relationship understanding

3. **Cross-Chunk Difficulty Progression**
   - Term relationships (Level 1-2)
   - Process integration (Level 3)
   - System connections (Level 4-5)
   - Complex interactions (Level 5)

4. **Multi-Hop Clarification Types**
   - Terms: Cross-context vocabulary
   - Processes: Multi-step sequences
   - Relationships: Inter-chunk connections
   - Context: Background integration
   - References: Cross-chunk mentions
   - Systems: Complex interactions

5. **Cross-Reference Quality**
   - Clear connection explanation
   - Integrated answers
   - Multi-chunk support
   - Enhanced relationship understanding