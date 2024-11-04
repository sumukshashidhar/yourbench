# Document-Based Edge Case Question Generation

You will receive a document extract and optionally a summary. Your task is to generate high-quality edge case questions that test understanding of boundary conditions, exceptions, and special cases mentioned or implied in the provided text extract.

## Core Principles

1. **Text-Based Boundaries**
   - Edge cases must be from text
   - Exceptions must be mentioned
   - Special conditions noted
   - Boundary conditions clear

2. **Question Diversity**
   - Exception cases
   - Boundary conditions
   - Special circumstances
   - Rule limitations
   - Extreme cases
   - Corner conditions

3. **Question Quality**
   - Clear edge identification
   - Text-supported exception
   - Valid special case
   - Meaningful boundary

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

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
        default=QuestionType.EDGE_CASE,
        description="Question type (edge-case)"
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

1. **Edge Identification**
   - Spot exceptions
   - Find boundaries
   - Note limitations
   - Identify special cases

2. **Case Development**
   - Define clear conditions
   - Map implications
   - Consider interactions
   - Verify text support

3. **Question Formation**
   - Present edge clearly
   - Request understanding
   - Enable explanation
   - Guide analysis

4. **Quality Verification**
   - Check text basis
   - Verify validity
   - Confirm clarity
   - Test usefulness

## Examples

### Example 1: Exception Case (Easy)

```json
{
    "document_extract_analysis": "The text explains grammar rules about 'i before e' in English spelling, noting specific exceptions.",
    "testable_concepts": [
        "spelling rules",
        "exceptions",
        "word patterns"
    ],
    "potential_question_directions": [
        "What specific examples illustrate the exceptions to the 'i before e' rule?",
        "In what ways does the text clarify the relationship between the 'i before e' rule and its exceptions?",
        "How do the exceptions mentioned in the text challenge the general spelling rule?"
    ],
    "best_direction": "What specific examples illustrate the exceptions to the 'i before e' rule?",
    "comprehension_type": "exception_case",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "The rule states 'i before e except after c'.",
        "However, this rule doesn't apply to words like 'weird' and 'height'.",
        "These exceptions occur mainly in words with the 'eigh' sound."
    ],
    "quote_context": "The quotes establish the rule and its exceptions.",
    "kind": "edge-case",
    "question": "Based on the text's explanation, why doesn't the 'i before e' rule apply to the word 'height', and what pattern does this exception follow?",
    "answer": "According to the text, 'height' is an exception because it contains the 'eigh' sound pattern, which regularly breaks the general rule",
    "reasoning": "The question tests understanding of a specific exception pattern mentioned in the text.",
    "difficulty": 2,
    "difficulty_justification": "Requires identifying a simple exception pattern explicitly stated in the text."
}
```

### Example 2: Boundary Condition (Medium)

```json
{
    "document_extract_analysis": "The passage describes water's behavior at different temperatures, including phase changes.",
    "testable_concepts": [
        "phase transitions",
        "temperature boundaries",
        "state changes"
    ],
    "potential_question_directions": [
        "What specific conditions allow water to transition between different phases?",
        "In what ways does the text clarify the relationship between temperature and water's phase changes?",
        "What unique characteristics define the boundary conditions for water's state changes?"
    ],
    "best_direction": "What specific conditions allow water to transition between different phases?",
    "comprehension_type": "boundary_condition",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Water typically freezes at 0°C at standard pressure.",
        "However, super-cooled water can remain liquid below 0°C under specific conditions.",
        "This metastable state occurs when water is very pure and undisturbed."
    ],
    "quote_context": "The quotes describe standard freezing and an exception case.",
    "kind": "edge-case",
    "question": "According to the text, under what specific boundary conditions can water exist below its normal freezing point while remaining liquid?",
    "answer": "The text indicates that super-cooled water can remain liquid below 0°C when it's very pure and undisturbed",
    "reasoning": "Tests understanding of specific conditions that create an exception to normal freezing behavior.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding multiple conditions that create a boundary case."
}
```

### Example 3: System Boundary (Very Hard)

```json
{
    "document_extract_analysis": "The text discusses ecosystem carrying capacity and population dynamics under stress conditions.",
    "testable_concepts": [
        "carrying capacity",
        "population limits",
        "system stress"
    ],
    "potential_question_directions": [
        "What specific factors influence the carrying capacity of an ecosystem?",
        "In what ways does the text illustrate the relationship between population dynamics and ecosystem stress?",
        "How do the described conditions affect the stability of an ecosystem's population?"
    ],
    "best_direction": "What specific factors influence the carrying capacity of an ecosystem?",
    "comprehension_type": "system_boundary",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Ecosystems typically maintain stable populations within carrying capacity.",
        "Under extreme resource limitation, population dynamics become chaotic.",
        "Recovery may be impossible if more than 60% of keystone species are lost.",
        "System collapse occurs when multiple stress factors combine."
    ],
    "quote_context": "The quotes describe system behavior at various boundaries.",
    "kind": "edge-case",
    "question": "Based on the text, what combination of conditions marks the boundary between ecosystem stress and irreversible system collapse?",
    "answer": "The text indicates collapse occurs with both keystone species loss exceeding 60% and multiple stress factors present, marking the boundary of recovery possibility",
    "reasoning": "Tests understanding of multiple interacting conditions that create a system boundary.",
    "difficulty": 5,
    "difficulty_justification": "Requires synthesizing multiple boundary conditions and understanding their interactions at a system level."
}
```

## Common Pitfalls to Avoid

1. **Invented Exceptions**
   ❌ "What if gravity stopped working?"
   ✅ "Under what conditions does the text say this rule doesn't apply?"

2. **External Cases**
   ❌ "What about cases not mentioned?"
   ✅ "What exception does the text specifically note?"

3. **Unclear Boundaries**
   ❌ "What might happen in extreme cases?"
   ✅ "What specific conditions create the boundary case described?"

4. **Invalid Combinations**
   ❌ "What if everything went wrong?"
   ✅ "What specific combination of factors creates this edge case?"

## Output Requirements

1. Generate 3-5 edge case questions per text extract
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure all cases are text-based
4. Include clear boundary conditions
5. Provide specific exception criteria
6. Scale complexity appropriately

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Easy/Exception)
    },
    {
        // Question 2 (Medium/Boundary)
    },
    {
        // Question 3 (Hard/System)
    },
    // ...
]
</generated_questions>
```

## Additional Guidelines

1. **Edge Case Selection**
   - Find explicit exceptions
   - Identify clear boundaries
   - Note special conditions
   - Map system limits

2. **Case Development**
   - Define specific conditions
   - Support with text
   - Explain implications
   - Show interactions

3. **Difficulty Progression**
   - Simple exceptions (Level 1-2)
   - Complex boundaries (Level 3)
   - Multiple conditions (Level 4)
   - System boundaries (Level 5)

4. **Edge Case Types**
   - Exceptions to rules
   - Boundary conditions
   - Special circumstances
   - System limits
   - Condition combinations
   - Corner cases

5. **Response Evaluation**
   - Clear conditions
   - Text support
   - Valid exception
   - Proper boundary