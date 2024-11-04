# Document-Based Counterfactual Question Generation

You will receive a document extract and a summary. Your task is to generate high-quality counterfactual questions that explore alternate scenarios based on changing specific elements from the provided text extract.

## Core Principles

1. **Text-Based Alterations**
   - Each question must modify a specific fact/event from the text
   - Answers must follow logically from text relationships
   - Changes must be clearly defined
   - Reasoning must use text evidence

2. **Question Diversity**
   - Event modifications
   - Character/entity changes
   - Decision alterations
   - Timing variations
   - Condition changes
   - Process modifications

3. **Question Quality**
   - Clear modification of text element
   - Logical chain of consequences
   - Grounded in text relationships
   - Plausible alternative scenarios

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

class QuestionType(str, Enum):
    COUNTERFACTUAL = "counterfactual"  # Questions exploring alternative scenarios

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Simple direct change
    EASY = 2         # Basic alternative scenario
    MEDIUM = 3       # Multi-step consequences
    HARD = 4         # Complex chain of effects
    VERY_HARD = 5    # System-wide implications

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
    text_based: bool = Field(..., description="Changes based on text elements")
    no_tricks: bool = Field(..., description="Avoids unrealistic scenarios")

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
        default=QuestionType.COUNTERFACTUAL,
        description="Question type (counterfactual)"
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

1. **Element Identification**
   - Identify key changeable elements
   - Note critical decision points
   - Map causal relationships
   - Identify system dependencies

2. **Change Mapping**
   - Define clear modifications
   - Trace impact chains
   - Consider system effects
   - Evaluate plausibility

3. **Question Formation**
   - Specify clear changes
   - Follow logical consequences
   - Maintain text relationships
   - Consider scope of impact

4. **Quality Verification**
   - Check change clarity
   - Verify logical flow
   - Confirm text basis
   - Test plausibility

## Examples

### Example 1: Event Change (Easy)

```json
{
    "document_extract_analysis": "The text describes how Alexander Fleming discovered penicillin when he noticed mold contaminating his bacterial cultures in 1928.",
    "testable_concepts": [
        "accidental discovery",
        "observation importance",
        "research methods"
    ],
    "potential_question_directions": [
        "What might have occurred if Fleming had chosen to discard the contaminated petri dish without further investigation?",
        "In what ways does the text highlight the significance of observation in the scientific discovery process?",
        "What critical elements contributed to the serendipitous discovery of penicillin by Fleming?"
    ],
    "best_direction": "What might have occurred if Fleming had chosen to discard the contaminated petri dish without further investigation?",
    "comprehension_type": "event_change",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Fleming noticed that mold had contaminated one of his petri dishes.",
        "He observed that bacteria wouldn't grow near the mold.",
        "This accidental contamination led to the discovery of penicillin."
    ],
    "quote_context": "The quotes establish the accidental nature of the discovery and its significance.",
    "kind": "counterfactual",
    "question": "Based on the text's description of events, what would likely have happened if Fleming had immediately discarded the contaminated petri dish instead of examining it?",
    "answer": "The discovery of penicillin's antibacterial properties would have been missed at this time",
    "reasoning": "The text shows that Fleming's observation of the contaminated dish was crucial to discovering penicillin's properties. Discarding it would have eliminated this opportunity for discovery.",
    "difficulty": 2,
    "difficulty_justification": "Requires understanding a simple cause-effect relationship and its alternative."
}
```

### Example 2: Process Change (Medium)

```json
{
    "document_extract_analysis": "The passage describes how the Panama Canal uses a system of locks to raise and lower ships between sea level and Gatun Lake.",
    "testable_concepts": [
        "lock system operation",
        "water level management",
        "ship transit process"
    ],
    "potential_question_directions": [
        "What would be the consequences if Gatun Lake were at the same elevation as sea level?",
        "In what ways does the text illustrate the importance of gravity in the operation of the lock system?",
        "What factors are essential for the effective functioning of the Panama Canal's lock system?"
    ],
    "best_direction": "What would be the consequences if Gatun Lake were at the same elevation as sea level?",
    "comprehension_type": "process_change",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "The locks work like water elevators, raising ships 85 feet above sea level.",
        "Each lock chamber fills with water from Gatun Lake using gravity alone.",
        "The system requires no pumps, saving significant energy.",
        "Ships are raised and lowered in a series of three locks."
    ],
    "quote_context": "The quotes detail the gravity-based lock system and its operation.",
    "kind": "counterfactual",
    "question": "According to the text's description of the canal system, how would the operation of the Panama Canal be affected if Gatun Lake were at sea level instead of 85 feet above?",
    "answer": "The gravity-based lock system would not function as no elevation difference would exist to move water naturally",
    "reasoning": "The text emphasizes that the system works by gravity due to Gatun Lake's elevation. Without this height difference, the fundamental mechanism would fail.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding the relationship between elevation, gravity, and system function to trace consequences."
}
```

### Example 3: System Impact (Very Hard)

```json
{
    "document_extract_analysis": "The text explains how the Gulf Stream affects climate patterns across the North Atlantic region.",
    "testable_concepts": [
        "ocean current patterns",
        "climate influence",
        "temperature regulation"
    ],
    "potential_question_directions": [
        "What would be the impact on Northern European climate if the Gulf Stream transported cold water instead of warm water?",
        "Which factors are essential in determining the Gulf Stream's effect on European climate?",
        "In what ways does the temperature of the Gulf Stream influence atmospheric circulation in the North Atlantic?"
    ],
    "best_direction": "What would be the impact on Northern European climate if the Gulf Stream transported cold water instead of warm water?",
    "comprehension_type": "system_impact",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "The Gulf Stream carries warm water from the Caribbean to Northern Europe.",
        "This warm current keeps European coastal temperatures 5-8°C warmer than similar latitudes.",
        "The temperature difference drives atmospheric circulation patterns.",
        "These patterns influence rainfall and storm systems across the region."
    ],
    "quote_context": "The quotes establish the Gulf Stream's role in climate regulation.",
    "kind": "counterfactual",
    "question": "Based on the text's description of the Gulf Stream system, how would Northern European climate patterns change if the Gulf Stream carried cold water instead of warm water?",
    "answer": "European coastal temperatures would drop 5-8°C, altering atmospheric circulation and changing rainfall and storm patterns across the region",
    "reasoning": "The text describes how the warm water directly affects temperatures, which drive atmospheric patterns. Reversing the temperature effect would impact the entire connected system.",
    "difficulty": 5,
    "difficulty_justification": "Requires understanding complex system interactions and tracing multiple levels of consequences."
}
```

## Common Pitfalls to Avoid

1. **Unrealistic Changes**
   ❌ "What if gravity didn't exist in the Panama Canal?"
   ✅ "What if Gatun Lake were at sea level?"

2. **Ungrounded Speculation**
   ❌ "What if Fleming had different career aspirations?"
   ✅ "What if Fleming had discarded the contaminated sample?"

3. **External Knowledge Dependency**
   ❌ "What if modern antibiotics existed in 1928?"
   ✅ "What if Fleming hadn't noticed the mold's effect?"

4. **Overly Broad Changes**
   ❌ "What if oceans didn't exist?"
   ✅ "What if the Gulf Stream carried cold water instead of warm?"

## Output Requirements

1. Generate 3-5 counterfactual questions per text extract
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure changes are specific and text-based
4. Include at least one system-level change
5. All reasoning must use text relationships
6. Changes must be plausible and meaningful

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Easy/Event Change)
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

1. **Change Selection**
   - Choose significant elements
   - Ensure clear modification
   - Consider impact scope
   - Maintain plausibility

2. **Consequence Tracing**
   - Follow logical chains
   - Use text relationships
   - Consider multiple effects
   - Maintain system consistency

3. **Difficulty Scaling**
   - Simple direct changes (Level 1-2)
   - Process modifications (Level 3)
   - System-wide impacts (Level 4-5)
   - Complex interactions (Level 5)

4. **Counterfactual Types**
   - Event: Changed occurrences
   - Decision: Alternative choices
   - Timing: Different sequences
   - Condition: Modified circumstances
   - Process: Alternative methods
   - System: Broad changes

5. **Response Evaluation**
   - Clear change specification
   - Logical consequence chain
   - Text-based reasoning
   - Plausible outcomes