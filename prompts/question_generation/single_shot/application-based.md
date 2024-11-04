# Document-Based Application Question Generation

You will receive a document extract and optionally a summary. Your task is to generate high-quality application questions that test the ability to apply principles, concepts, or processes from the text to new situations.

## Core Principles

1. **Text-Based Application**
   - Application must use text principles
   - New scenarios must be relatable
   - No external knowledge required
   - Clear connection to text content

2. **Question Diversity**
   - Principle application
   - Process adaptation
   - Model use
   - Problem solving
   - Strategy application
   - Method transfer

3. **Question Quality**
   - Clear scenario
   - Obvious text connection
   - Reasonable application
   - Meaningful transfer

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

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

1. **Principle Identification**
   - Identify key concepts
   - Note processes
   - Map methods
   - Understand systems

2. **Scenario Development**
   - Create relatable situations
   - Ensure principle fit
   - Enable application
   - Maintain relevance

3. **Question Formation**
   - Present clear scenario
   - Request specific application
   - Guide thought process
   - Enable demonstration

4. **Quality Verification**
   - Check text connection
   - Verify applicability
   - Confirm clarity
   - Test usefulness

## Examples

### Example 1: Principle Application (Easy)

```json
{
    "document_extract_analysis": "The text explains the principle of supply and demand, showing how prices change based on market conditions.",
    "testable_concepts": [
        "price mechanism",
        "market balance",
        "economic responses"
    ],
    "potential_question_directions": [
        "In what ways does a surge in umbrella demand influence pricing during a rainy season?",
        "What impact does a sudden influx of umbrella supply have on market prices?",
        "How do fluctuations in demand and supply interact to determine umbrella pricing?"
    ],
    "best_direction": "In what ways does a surge in umbrella demand influence pricing during a rainy season?",
    "comprehension_type": "principle_application",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "When supply exceeds demand, prices typically fall.",
        "Higher prices tend to reduce demand while encouraging more supply.",
        "Markets seek equilibrium through price adjustments."
    ],
    "quote_context": "The quotes establish basic supply-demand principles.",
    "kind": "application",
    "question": "Using the supply and demand principles described in the text, what would likely happen to umbrella prices during an unexpected rainy season?",
    "answer": "Based on the text's principles, increased demand for umbrellas would drive prices up, which might then encourage more supply",
    "reasoning": "Applies the text's price mechanism principles to a specific scenario.",
    "difficulty": 2,
    "difficulty_justification": "Straightforward application of basic market principles to a simple scenario."
}
```

### Example 2: Process Adaptation (Medium)

```json
{
    "document_extract_analysis": "The text details scientific method steps: observation, hypothesis formation, testing, and conclusion drawing.",
    "testable_concepts": [
        "scientific method",
        "research process",
        "investigation steps"
    ],
    "potential_question_directions": [
        "In what ways can the scientific method be utilized to explore the reasons behind varying success rates of houseplants in an office environment?",
        "What essential steps are necessary for employing the scientific method to examine plant growth across different settings?",
        "How can the scientific method enhance our understanding of the variables affecting plant growth in an office context?"
    ],
    "best_direction": "In what ways can the scientific method be utilized to explore the reasons behind varying success rates of houseplants in an office environment?",  
    "comprehension_type": "process_adaptation",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Scientists begin by observing a phenomenon.",
        "They form a hypothesis to explain their observations.",
        "Controlled experiments test the hypothesis.",
        "Results lead to conclusions and potential theory revision."
    ],
    "quote_context": "The quotes outline the scientific method process.",
    "kind": "application",
    "question": "How could you apply the scientific method steps described in the text to investigate why some houseplants thrive better than others in an office environment?",
    "answer": "Following the text's process: observe different plant success rates, form hypothesis about factors (light, water, etc.), test through controlled comparisons, and draw conclusions from results",
    "reasoning": "Adapts the formal scientific method to a practical investigation.",
    "difficulty": 3,
    "difficulty_justification": "Requires adapting a structured process to a new context while maintaining key elements."
}
```

### Example 3: System Application (Very Hard)

```json
{
    "document_extract_analysis": "The passage explains ecosystem feedback loops, particularly in predator-prey relationships.",
    "testable_concepts": [
        "feedback systems",
        "population dynamics",
        "ecosystem balance"
    ],
    "potential_question_directions": [
        "In what ways could the introduction of a new predator species alter the dynamics of predator-prey interactions on a remote island?",
        "What ecological impacts might arise from adding a new predator species to an established ecosystem?",
        "How could the introduction of a new predator species disrupt the existing balance of predator-prey relationships in a specific ecosystem?"
    ],
    "best_direction": "In what ways could the introduction of a new predator species alter the dynamics of predator-prey interactions on a remote island?",
    "comprehension_type": "theory_application",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Predator populations rise when prey is abundant.",
        "Increased predation reduces prey populations.",
        "Fewer prey leads to predator decline.",
        "Reduced predation allows prey recovery, completing the cycle."
    ],
    "quote_context": "The quotes describe ecosystem feedback mechanisms.",
    "kind": "application",
    "question": "Using the ecosystem feedback principles described in the text, how might introducing a new predator species affect an isolated island's existing predator-prey relationships?",
    "answer": "Based on the text's principles, the new predator would increase total predation, potentially disrupting existing cycles: faster prey decline, affecting original predators, leading to complex feedback loops until new balance emerges",
    "reasoning": "Applies understanding of feedback mechanisms to a complex system modification.",
    "difficulty": 5,
    "difficulty_justification": "Requires applying system-level understanding to predict complex interactions and multiple feedback effects."
}
```

## Common Pitfalls to Avoid

1. **Unrealistic Scenarios**
   ❌ "Apply ecosystem principles to Mars colonization"
   ✅ "Apply ecosystem principles to a new nature reserve"

2. **External Knowledge Requirement**
   ❌ "Use quantum physics to explain chemistry"
   ✅ "Use the text's principles to analyze a similar situation"

3. **Overly Complex Applications**
   ❌ "Apply to every possible scenario"
   ✅ "Apply to a specific, relevant situation"

4. **Disconnected Scenarios**
   ❌ "Apply economic principles to space travel"
   ✅ "Apply economic principles to a new market situation"

## Output Requirements

1. Generate 3-5 application questions per text extract
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure clear connection to text principles
4. Include realistic scenarios
5. Provide valid application paths
6. Scale difficulty appropriately

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Easy/Principle)
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

1. **Application Design**
   - Choose relevant principles
   - Create realistic scenarios
   - Enable clear application
   - Maintain text connection

2. **Scenario Development**
   - Use relatable situations
   - Scale complexity appropriately
   - Ensure principle relevance
   - Enable demonstration

3. **Difficulty Progression**
   - Simple application (Level 1-2)
   - Process adaptation (Level 3)
   - Complex application (Level 4)
   - System application (Level 5)

4. **Application Types**
   - Principles to scenarios
   - Processes to new contexts
   - Models to situations
   - Methods to problems
   - Theories to cases

5. **Response Evaluation**
   - Clear application path
   - Principle adherence
   - Reasonable outcome
   - Valid demonstration