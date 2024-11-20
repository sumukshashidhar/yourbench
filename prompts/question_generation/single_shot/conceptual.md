# Document-Based Conceptual Question Generation

You will receive a document extract and optionally a summary. Your task is to generate high-quality conceptual questions that probe understanding of core ideas, principles, and relationships presented in the text.

## Core Principles

1. **Concept-Based Inquiry**
   - Questions must target key concepts from the text
   - Answers should demonstrate conceptual understanding
   - No mere fact recall
   - Focus on principles and relationships

2. **Question Diversity**
   - Core principle understanding
   - Concept application
   - Relationship comprehension
   - Framework understanding
   - Theory explanation
   - Model comprehension

3. **Question Quality**
   - Clear conceptual focus
   - Tests understanding not memory
   - Requires principle application
   - Explores key relationships

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

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
    Represents a structured QA pair for document comprehension testing.
    """
    # Analysis Fields
    document_extract_analysis: str = Field(
        ...,
        description="Analysis of the key points and structure of the extract",
        min_length=50
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
        default=QuestionType.CONCEPTUAL,
        description="Question type (conceptual)"
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

1. **Concept Identification**
   - Identify key principles
   - Map conceptual relationships
   - Note theoretical frameworks
   - Identify core models

2. **Understanding Assessment**
   - Define conceptual focus
   - Map principle applications
   - Consider relationships
   - Evaluate depth

3. **Question Formation**
   - Target key concepts
   - Require understanding
   - Test relationships
   - Explore applications

4. **Quality Verification**
   - Check concept clarity
   - Verify understanding focus
   - Confirm text basis
   - Test depth

## Examples

### Example 1: Process Analysis (Easy)
```json
{
    "document_extract_analysis": "The text describes photosynthesis as a process where plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
    "testable_concepts": [
        "energy conversion",
        "reactants and products",
        "cellular processes"
    ],
    "potential_question_directions": [
        "How do plants transform solar energy into chemical energy?",
        "What role does each reactant play in photosynthesis?",
        "Why is photosynthesis essential for life on Earth?"
    ],
    "best_direction": "How do plants transform solar energy into chemical energy?",
    "comprehension_type": "process_analysis",
    "kind": "conceptual",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "What makes photosynthesis an energy transformation process rather than just a chemical reaction?",
    "answer": "Photosynthesis converts light energy from the sun into stored chemical energy in glucose molecules, representing a fundamental energy transformation in nature",
    "reasoning": "The question tests understanding of energy conversion principles and distinguishes between simple chemical reactions and energy transformation processes",
    "difficulty": 2,
    "difficulty_justification": "Requires basic understanding of energy transformation but concepts are clearly presented in the text",
    "supporting_quotes": [
        "Plants capture sunlight through chlorophyll molecules in their leaves",
        "The process converts CO2 and H2O into glucose (C6H12O6) and oxygen (O2)",
        "This energy transformation is fundamental to life on Earth"
    ],
    "quote_context": "These quotes establish the key components of photosynthesis, showing both the inputs and outputs while emphasizing its role as an energy transformation process rather than just a chemical reaction"
}
```

### Example 2: System Analysis (Medium)
```json
{
    "document_extract_analysis": "The passage explains how the immune system recognizes and responds to pathogens through multiple coordinated mechanisms.",
    "testable_concepts": [
        "immune response",
        "pathogen recognition",
        "cellular coordination"
    ],
    "potential_question_directions": [
        "How do different components of the immune system work together?",
        "What triggers the initial immune response?",
        "Why is the immune response both general and specific?"
    ],
    "best_direction": "How do different components of the immune system work together?",
    "comprehension_type": "system_analysis",
    "kind": "conceptual",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How does the coordination between different immune cells enhance the body's defense against pathogens?",
    "answer": "Different immune cells communicate and coordinate their actions, with some identifying threats, others signaling danger, and specialized cells mounting targeted responses",
    "reasoning": "Tests understanding of system complexity and cellular interaction in immune response",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding multiple components and their interactions in a complex system",
    "supporting_quotes": [
        "White blood cells recognize specific molecular patterns on pathogen surfaces",
        "Dendritic cells present antigens to T-cells, triggering a targeted immune response",
        "Cytokines released by infected cells alert nearby immune cells to the threat"
    ],
    "quote_context": "These quotes demonstrate the coordinated nature of immune response, showing how different cell types communicate and work together to mount an effective defense"
}
```

### Example 3: Relationship Comprehension (Hard)
```json
{
    "document_extract_analysis": "The text explores the relationship between climate change, ocean acidification, and marine ecosystem collapse.",
    "testable_concepts": [
        "carbon cycle",
        "ecosystem interdependence",
        "environmental feedback loops"
    ],
    "potential_question_directions": [
        "How do changes in ocean pH affect marine food webs?",
        "What connects atmospheric carbon levels to marine life survival?",
        "Why do small pH changes have large ecosystem effects?"
    ],
    "best_direction": "How do changes in ocean pH affect marine food webs?",
    "comprehension_type": "relationship_comprehension",
    "kind": "conceptual",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "Why does ocean acidification have cascading effects throughout marine ecosystems rather than just affecting single species?",
    "answer": "Ocean acidification disrupts fundamental biological processes at the base of food webs, creating chain reactions that affect all dependent species through their ecological connections",
    "reasoning": "Tests understanding of complex ecological relationships and system-wide effects",
    "difficulty": 4,
    "difficulty_justification": "Requires analysis of multiple interconnected ecological relationships and their consequences",
    "supporting_quotes": [
        "As CO2 levels rise, ocean pH decreases, making it difficult for calcifying organisms to build shells",
        "Pteropods, key prey species for many marine organisms, are particularly vulnerable to acidification",
        "The loss of these base food chain organisms creates ripple effects throughout marine ecosystems"
    ],
    "quote_context": "These quotes establish the chain of causation from ocean acidification to ecosystem-wide effects, highlighting how the impact on fundamental species affects the entire food web"
}
```

### Example 4: Evidence Synthesis (Very Hard)
```json
{
    "document_extract_analysis": "The text discusses how quantum entanglement challenges classical physics principles and enables quantum computing.",
    "testable_concepts": [
        "quantum entanglement",
        "classical physics limitations",
        "quantum information theory"
    ],
    "potential_question_directions": [
        "How does entanglement enable quantum advantages?",
        "Why can't classical physics explain entanglement?",
        "What makes quantum information fundamentally different?"
    ],
    "best_direction": "How does entanglement enable quantum advantages?",
    "comprehension_type": "evidence_synthesis",
    "kind": "conceptual",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How does quantum entanglement's violation of classical physics principles create new possibilities for information processing?",
    "answer": "Entanglement allows quantum systems to exist in multiple states simultaneously and maintain instantaneous correlations, enabling parallel processing and secure communication impossible in classical systems",
    "reasoning": "Requires synthesizing concepts from quantum mechanics, information theory, and classical physics limitations",
    "difficulty": 5,
    "difficulty_justification": "Demands integration of multiple complex concepts and understanding of paradigm-shifting principles",
    "supporting_quotes": [
        "Quantum entanglement allows particles to maintain instantaneous correlations regardless of distance",
        "Unlike classical bits, quantum bits can exist in multiple states simultaneously through superposition",
        "These quantum properties enable exponential increases in computational power for certain algorithms"
    ],
    "quote_context": "These quotes connect the fundamental quantum mechanical principle of entanglement to its practical applications in quantum computing, showing how it transcends classical physics limitations"
}
```

### Example 5: Mechanism Understanding (Medium)
```json
{
    "document_extract_analysis": "The text explains how neurons use action potentials and neurotransmitters to communicate information.",
    "testable_concepts": [
        "action potential generation",
        "synaptic transmission",
        "neural coding"
    ],
    "potential_question_directions": [
        "How do electrical and chemical signals work together in neural communication?",
        "What determines the strength of synaptic transmission?",
        "Why is neural communication both electrical and chemical?"
    ],
    "best_direction": "How do electrical and chemical signals work together in neural communication?",
    "comprehension_type": "mechanism_understanding",
    "kind": "conceptual",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How does the conversion between electrical and chemical signals enable precise neural communication?",
    "answer": "Neurons convert electrical action potentials into chemical neurotransmitter release, allowing for signal modification and specific targeting of recipient cells",
    "reasoning": "Tests understanding of the dual nature of neural signaling and its functional significance",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding of multiple mechanisms and their integration in neural communication",
    "supporting_quotes": [
        "Action potentials propagate electrically along the neuron's axon",
        "At synapses, electrical signals trigger the release of chemical neurotransmitters",
        "The amount of neurotransmitter released determines the strength of the signal received by the next neuron"
    ],
    "quote_context": "These quotes outline the sequence of neural communication, showing the transition from electrical to chemical signaling and how this enables signal modulation"
}
```

## Common Pitfalls to Avoid

1. **Fact Recall Questions**
   ❌ "What is the atomic number?"
   ✅ "How does atomic number determine element identity?"

2. **Surface Understanding**
   ❌ "List the parts of an atom."
   ✅ "How do different atomic components determine element properties?"

3. **External Knowledge**
   ❌ "Compare this to modern atomic theory."
   ✅ "How does this model explain atomic identity?"

4. **Oversimplification**
   ❌ "Is memory chemical or structural?"
   ✅ "How do chemical and structural changes interact in memory formation?"

## Output Requirements

1. Generate 3-5 conceptual questions per text extract
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure questions test understanding not memory
4. Include at least one synthesis question
5. All concepts must be from the text
6. Questions should explore relationships and applications

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1
    },
    {
        // Question 2 
    },
    {
        // Question 3 
    },
    // ...
]
</generated_questions>
```

## Additional Guidelines

1. **Concept Selection**
   - Choose fundamental principles
   - Focus on key relationships
   - Target core models
   - Include theoretical frameworks

2. **Understanding Assessment**
   - Test principle application
   - Explore relationships
   - Examine model use
   - Require synthesis

3. **Difficulty Progression**
   - Basic principles (Level 1-2)
   - Model application (Level 3)
   - Complex relationships (Level 4)
   - Deep synthesis (Level 5)

4. **Conceptual Types**
   - Principles: Core ideas and rules
   - Models: Frameworks and structures
   - Relationships: Connections between concepts
   - Applications: Using concepts
   - Synthesis: Combining ideas

5. **Response Evaluation**
   - Clear conceptual understanding
   - Principle application
   - Relationship comprehension
   - Multiple concept integration