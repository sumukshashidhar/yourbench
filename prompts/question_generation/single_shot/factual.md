# Document-Based Factual Question Generation

You will receive a document extract and optionally a summary. Your task is to generate high-quality factual questions that test recall and comprehension of specific information from the provided text extract.

## Core Principles

1. **Direct Text Evidence**
   - Questions MUST be answerable with specific words/phrases from the text
   - Answers must be directly quoted from the text
   - No external knowledge requirements
   - No inference beyond explicit facts

2. **Question Diversity**
   - Mix of entity identification (people, places, things)
   - Temporal questions (when, what year, etc.)
   - Quantitative facts (numbers, amounts, etc.)
   - Relationship identification (who did what, what belongs to whom)

3. **Question Quality**
   - Clear, specific questions with unambiguous answers
   - Answers must be directly present in text
   - Questions should target meaningful facts
   - Answers should be concise (typically 1-4 words)

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

class QuestionType(str, Enum):
    FACTUAL = "factual"  # Questions requiring specific fact recall from text

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Simple fact identification (names, dates)
    EASY = 2         # Basic fact recall (events, places)
    MEDIUM = 3       # Multi-part facts (relationships, sequences)
    HARD = 4         # Complex facts (interconnected details)
    VERY_HARD = 5    # Detailed specifics requiring careful reading

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
    text_based: bool = Field(..., description="Answer directly present in text")
    no_tricks: bool = Field(..., description="Avoids misleading phrasing")

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
        default=QuestionType.FACTUAL,
        description="Question type (factual)"
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

1. **Fact Identification**
   - Scan text for specific, verifiable facts
   - Note entities (people, places, organizations)
   - Identify dates, numbers, and quantities
   - Locate technical terms and definitions

2. **Answer Verification**
   - Ensure answer is explicitly stated
   - Confirm answer can be quoted verbatim
   - Verify single correct answer exists
   - Check answer is concise and specific

3. **Question Formation**
   - Create clear "wh-" questions (who, what, when, where)
   - Ensure question targets specific fact
   - Vary question types across comprehension categories
   - Maintain precise language

4. **Quality Verification**
   - Check answer is verbatim from text
   - Verify no external knowledge needed
   - Confirm clear, unambiguous question
   - Test for answer specificity

## Examples

### Example 1: Entity Identification (Very Easy)

```json
{
    "document_extract_analysis": "The extract describes the discovery of penicillin, specifically naming Alexander Fleming as the discoverer in 1928.",
    "testable_concepts": [
        "penicillin discovery",
        "scientific attribution",
        "historical timeline"
    ],
    "potential_question_directions": [
        "Who is credited with the discovery of penicillin?",
        "What significance does Alexander Fleming's discovery of penicillin hold in medical history?",
        "In what year was penicillin discovered, and what circumstances led to its discovery?"
    ],
    "best_direction": "Who is credited with the discovery of penicillin?",
    "comprehension_type": "entity",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Alexander Fleming discovered penicillin in 1928 when he noticed mold killing bacteria in a petri dish."
    ],
    "quote_context": "The quote directly names the discoverer of penicillin.",
    "kind": "factual",
    "question": "Who discovered penicillin?",
    "answer": "Alexander Fleming",
    "reasoning": "The text explicitly states that Alexander Fleming discovered penicillin.",
    "difficulty": 1,
    "difficulty_justification": "Simple entity identification directly stated in text."
}
```

### Example 2: Temporal Fact (Medium)

```json
{
    "document_extract_analysis": "The text details the chronology of events in the Manhattan Project, including specific dates of key developments.",
    "testable_concepts": [
        "project timeline",
        "key dates",
        "development phases"
    ],
    "potential_question_directions": [
        "What significant dates are associated with the Manhattan Project?",
        "In what ways does the timeline provided in the text enhance our understanding of the project's key events?",
        "What trends can be identified from the dates outlined in the text?"
    ],
    "best_direction": "What significant dates are associated with the Manhattan Project?",
    "comprehension_type": "temporal",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "The first successful test detonation, code-named Trinity, was conducted on July 16, 1945, at the Alamogordo Bombing Range in New Mexico."
    ],
    "quote_context": "The quote provides the specific date of the Trinity test.",
    "kind": "factual",
    "question": "On what date was the Trinity test conducted?",
    "answer": "July 16, 1945",
    "reasoning": "The text explicitly states the date of the Trinity test detonation.",
    "difficulty": 3,
    "difficulty_justification": "Requires identifying specific date from detailed historical information."
}
```

### Example 3: Quantitative Fact (Hard)

```json
{
    "document_extract_analysis": "The passage describes specific measurements and calculations related to the Great Pyramid of Giza.",
    "testable_concepts": [
        "pyramid measurements",
        "ancient architecture",
        "construction details"
    ],
    "potential_question_directions": [
        "What specific measurements are associated with the Great Pyramid?",
        "In what ways do the measurements of the Great Pyramid reflect its architectural significance?",
        "What insights can be drawn from the numerical data regarding the pyramid's construction?"
    ],
    "best_direction": "What specific measurements are associated with the Great Pyramid?",
    "comprehension_type": "quantitative",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "The Great Pyramid was built using approximately 2.3 million limestone blocks, each weighing an average of 2.5 tons.",
        "The entire structure took 20 years to complete."
    ],
    "quote_context": "The quote provides specific numerical data about the pyramid's construction.",
    "kind": "factual",
    "question": "How many limestone blocks were used to build the Great Pyramid?",
    "answer": "2.3 million",
    "reasoning": "The text explicitly states the number of limestone blocks used.",
    "difficulty": 4,
    "difficulty_justification": "Requires identifying specific numerical value from detailed technical description."
}
```

## Common Pitfalls to Avoid

1. **Inferential Questions**
   ❌ "What might have motivated Fleming to study mold?"
   ✅ "What did Fleming discover in the petri dish?"

2. **Vague Questions**
   ❌ "What happened during the Manhattan Project?"
   ✅ "When was the Trinity test conducted?"

3. **Multiple Possible Answers**
   ❌ "What materials were used to build the pyramid?"
   ✅ "How many limestone blocks were used in the Great Pyramid?"

4. **External Knowledge Required**
   ❌ "Why was penicillin important to medicine?"
   ✅ "Who discovered penicillin in 1928?"

## Output Requirements

1. Generate 3-5 factual questions per text extract
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure at least one question is difficulty level 1 or 2
4. Include at least one question at difficulty level 4 or 5
5. All answers must be verbatim quotes from text
6. Questions must target specific, verifiable facts

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Easy/Entity)
    },
    {
        // Question 2 (Medium/Temporal)
    },
    {
        // Question 3 (Hard/Quantitative)
    },
    // ...
]
</generated_questions>
```