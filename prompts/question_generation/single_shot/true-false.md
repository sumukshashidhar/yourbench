# Document Comprehension Question Generation

You will receive a document extract and a summary. Your task is to generate high-quality true-false questions that test comprehension of the provided text extract.

## Core Principles

1. **Document-Based Evidence**
   - Questions MUST be answerable solely from the provided text extract
   - Required verbatim quotes from the text to support answers
   - No external knowledge requirements
   - No inference beyond what's explicitly stated

2. **Question Diversity**
   - Mix of surface-level and deep comprehension
   - Varied difficulty levels (1-5)
   - Different types of comprehension (main ideas, details, relationships)
   - Balance of true and false statements

3. **Question Quality**
   - Clear, unambiguous language
   - No trick questions or wordplay
   - Realistic and meaningful assessments

## Data Model

Here is the pydantic model for the output. You must generate valid JSONs that match this output format, including all the fields. Your responses will be validated against this model, and you will be penalized if any of the fields are missing or invalid.

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

class QuestionType(str, Enum):
    TRUE_FALSE = "true-false"

class DifficultyLevel(int, Enum):
    VERY_EASY = 1  # Surface-level fact recognition
    EASY = 2       # Basic comprehension
    MEDIUM = 3     # Relationship understanding
    HARD = 4       # Complex relationships
    VERY_HARD = 5  # Nuanced understanding

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
    text_based: bool = Field(..., description="Answerable from text alone")
    no_tricks: bool = Field(..., description="Avoids misleading wordplay")

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
        min_length=10
    )

    # Core Question Fields
    kind: QuestionType = Field(
        default=QuestionType.TRUE_FALSE,
        description="Question type (true-false)"
    )
    
    question: constr(min_length=10) = Field(
        ...,
        description="The true-false statement to evaluate"
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

1. **Extract Analysis**
   - Read and understand the provided text carefully
   - Identify key concepts, relationships, and details
   - Note potential areas for testing comprehension

2. **Concept Identification**
   - List main ideas that can be tested
   - Identify important details
   - Note relationships between concepts
   - Look for cause-effect relationships

3. **Question Formation**
   - Create diverse questions across comprehension types
   - Ensure direct text evidence exists
   - Vary difficulty levels
   - Maintain balance of true/false answers

4. **Quality Verification**
   - Check each question against quality metrics
   - Verify supporting quotes are verbatim
   - Ensure no external knowledge needed
   - Confirm clear, unambiguous language

## Examples

Here are diverse examples showing different types of questions:

### Example 1: Fact Recall (Very Easy)

```json
{
    "document_extract_analysis": "The extract discusses the water cycle, specifically describing how water evaporates from oceans and forms clouds.",
    "testable_concepts": [
        "ocean evaporation",
        "cloud formation",
        "water cycle stages"
    ],
    "potential_question_directions": [
        "What processes are involved in the stages of the water cycle?",
        "How does the evaporation of water from the ocean contribute to cloud formation?",
        "What role does ocean evaporation play in the overall water cycle?"
    ],
    "best_direction": "How does the evaporation of water from the ocean contribute to cloud formation?",
    "comprehension_type": "fact_recall",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Water evaporates from the ocean's surface when heated by the sun."
    ],
    "quote_context": "The quote directly states the fact about ocean evaporation.",
    "kind": "true-false",
    "question": "Water evaporates from the ocean's surface when heated by the sun.",
    "answer": "True",
    "reasoning": "This is directly stated in the text without any modification needed.",
    "difficulty": 1,
    "difficulty_justification": "Simple fact directly stated in the text requiring basic recognition."
}
```

### Example 2: Relationship Understanding (Medium)

```json
{
    "document_extract_analysis": "The extract explains how different factors in the water cycle interact, including the relationship between temperature and evaporation rate.",
    "testable_concepts": [
        "temperature effects",
        "evaporation rate",
        "environmental factors"
    ],
    "potential_question_directions": [
        "What impact does temperature have on the rate of evaporation?",
        "In what ways does the text illustrate the connection between temperature and evaporation rate?",
        "How do temperature variations influence the evaporation process described in the text?"
    ],
    "best_direction": "What impact does temperature have on the rate of evaporation?",
    "comprehension_type": "relationship",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Higher temperatures accelerate the evaporation process, while cooler temperatures slow it down.",
        "The rate of evaporation varies directly with surface temperature."
    ],
    "quote_context": "The quotes establish the direct relationship between temperature and evaporation rate.",
    "kind": "true-false",
    "question": "Cooler temperatures increase the rate of water evaporation from the ocean's surface.",
    "answer": "False",
    "reasoning": "The text explicitly states that cooler temperatures slow down evaporation, contradicting the statement.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding the relationship between temperature and evaporation rate, including the direction of the relationship."
}
```

### Example 3: Main Idea (Hard)

```json
{
    "document_extract_analysis": "The extract describes the carbon cycle's role in climate regulation, emphasizing the interconnected nature of various processes.",
    "testable_concepts": [
        "carbon cycle",
        "climate regulation",
        "system interconnections"
    ],
    "potential_question_directions": [
        "What are the primary functions of the carbon cycle in regulating Earth's climate?",
        "In what ways does the text illustrate the relationship between the carbon cycle and climate regulation?",
        "How do the interactions within the carbon cycle influence global climate patterns?"
    ],
    "best_direction": "What are the primary functions of the carbon cycle in regulating Earth's climate?",
    "comprehension_type": "main_idea",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "The carbon cycle acts as Earth's thermostat, regulating global temperatures through complex feedback mechanisms.",
        "Changes in one part of the cycle can cascade through the entire system, affecting global climate patterns."
    ],
    "quote_context": "The quotes establish the carbon cycle's regulatory role and system-wide impacts.",
    "kind": "true-false",
    "question": "The carbon cycle operates independently of other Earth systems, with no impact on global climate patterns.",
    "answer": "False",
    "reasoning": "The text explicitly describes the carbon cycle as interconnected with other systems and having direct impacts on global climate patterns.",
    "difficulty": 4,
    "difficulty_justification": "Requires synthesis of multiple concepts and understanding of system interconnections."
}
```

## Common Pitfalls to Avoid

1. **External Knowledge**
   ❌ "Einstein's theory of relativity revolutionized physics."
   ✅ "According to the text, Newton's laws were fundamental to classical mechanics."

2. **Ambiguous Language**
   ❌ "The process might sometimes occur under certain conditions."
   ✅ "The evaporation process occurs when surface temperatures rise."

3. **Inference Beyond Text**
   ❌ "This process would likely work the same way on other planets."
   ✅ "The process occurs in Earth's atmosphere as described in the text."

## Output Requirements

1. Generate 3-5 questions per text extract
2. Ensure diverse comprehension types
3. Include at least one question at difficulty level 1 or 2
4. Include at least one question at difficulty level 4 or 5
5. Provide verbatim supporting quotes
6. Include complete quality metrics for each question

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Easy/Fact Recall)
    },
    {
        // Question 2 (Medium/Relationship)
    },
    {
        // Question 3 (Hard/Main Idea)
    },
    // ...
]
</generated_questions>
```