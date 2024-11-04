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

### Example 1: Principle Understanding (Easy)

```json
{
    "document_extract_analysis": "The text explains the principle of natural selection, focusing on how environmental pressures lead to differential survival.",
    "testable_concepts": [
        "natural selection",
        "environmental pressure",
        "differential survival"
    ],
    "potential_question_directions": [
        "What role do environmental pressures play in shaping the traits of a population?",
        "How do variations in survival rates among organisms contribute to the process of natural selection?",
        "In what ways do specific traits enhance an organism's chances of survival in changing environments?"
    ],
    "best_direction": "What role do environmental pressures play in shaping the traits of a population?",
    "comprehension_type": "principle_understanding",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Natural selection occurs when environmental pressures favor certain traits.",
        "Organisms with advantageous traits are more likely to survive and reproduce.",
        "Over time, beneficial traits become more common in the population."
    ],
    "quote_context": "The quotes establish the basic principle of natural selection.",
    "kind": "conceptual",
    "question": "Based on the text's explanation, how does natural selection lead to changes in a population over time?",
    "answer": "Environmental pressures favor certain traits, leading to increased survival and reproduction of organisms with these traits, making them more common over time",
    "reasoning": "The text outlines the process of natural selection through environmental pressure, differential survival, and trait inheritance.",
    "difficulty": 2,
    "difficulty_justification": "Requires basic understanding of a fundamental principle explained in the text."
}
```

### Example 2: Model Comprehension (Medium)

```json
{
    "document_extract_analysis": "The passage explains the atomic model of matter, describing how different elements are determined by proton number.",
    "testable_concepts": [
        "atomic structure",
        "element definition",
        "atomic number"
    ],
    "potential_question_directions": [
        "In what way does the atomic number define the identity of an element?",
        "What effects do variations in neutron numbers have on the properties of an element?",
        "How is the atomic number connected to the arrangement of electrons in an atom?"
    ],
    "best_direction": "In what way does the atomic number define the identity of an element?",
    "comprehension_type": "model_comprehension",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Each element has a unique number of protons in its nucleus.",
        "This proton number, called the atomic number, defines the element.",
        "Atoms of the same element can have different numbers of neutrons.",
        "Electrons orbit the nucleus in specific energy levels."
    ],
    "quote_context": "The quotes outline the key aspects of atomic structure.",
    "kind": "conceptual",
    "question": "According to the text's model of atomic structure, what determines whether two atoms represent the same element?",
    "answer": "The number of protons (atomic number) in the nucleus determines the element, regardless of neutron count",
    "reasoning": "The text explicitly states that proton number defines the element, while allowing for variation in neutrons.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding the relationship between atomic structure and element identity."
}
```

### Example 3: Concept Synthesis (Very Hard)

```json
{
    "document_extract_analysis": "The text explains how memory formation involves both chemical and structural changes in neural networks.",
    "testable_concepts": [
        "synaptic plasticity",
        "neural networks",
        "memory formation"
    ],
    "potential_question_directions": [
        "In what ways do chemical changes at synapses initiate memory formation?",
        "What role does synaptic plasticity play in the process of memory retention?",
        "How do the structural adaptations in neural networks facilitate long-term memory storage?"
    ],
    "best_direction": "In what ways do chemical changes at synapses initiate memory formation?",
    "comprehension_type": "concept_synthesis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Memory formation begins with chemical changes at synapses.",
        "Repeated activation strengthens neural connections.",
        "Structural changes occur in neural networks over time.",
        "Both immediate chemical and long-term structural changes contribute to memory storage."
    ],
    "quote_context": "The quotes describe the multi-level process of memory formation.",
    "kind": "conceptual",
    "question": "Based on the text's explanation, how do chemical and structural changes work together in the formation of lasting memories?",
    "answer": "Initial chemical changes at synapses enable immediate memory formation, while repeated activation leads to structural changes in neural networks for long-term storage",
    "reasoning": "The text describes a two-level process where immediate chemical changes and long-term structural modifications work together in memory formation.",
    "difficulty": 5,
    "difficulty_justification": "Requires synthesizing multiple concepts and understanding their interaction over time."
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
        // Question 1 (Easy/Principle)
    },
    {
        // Question 2 (Medium/Model)
    },
    {
        // Question 3 (Hard/Synthesis)
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