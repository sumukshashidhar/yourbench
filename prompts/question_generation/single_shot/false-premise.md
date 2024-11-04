# Document-Based False Premise Question Generation

You will receive a document extract and optionally a summary. Your task is to generate high-quality false premise questions that present incorrect assumptions or assertions that contradict information in the provided text extract.

## Core Principles

1. **Text-Based Contradiction**
   - False premises must clearly contradict text
   - Correction must be supported by text
   - No external knowledge required
   - Clear evidence for correction

2. **Question Diversity**
   - Factual contradictions
   - Process inversions
   - Relationship errors
   - Temporal mistakes
   - Causal errors
   - Definition mistakes

3. **Question Quality**
   - Clear false premise
   - Obvious text contradiction
   - Evidence-based correction
   - Meaningful error

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

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

1. **Contradiction Identification**
   - Identify key facts
   - Note relationships
   - Map processes
   - Understand systems

2. **Premise Inversion**
   - Create clear contradictions
   - Ensure text evidence
   - Make errors identifiable
   - Plan corrections

3. **Question Formation**
   - Present false premise
   - Request correction
   - Enable evidence use
   - Support explanation

4. **Quality Verification**
   - Check contradiction clarity
   - Verify text evidence
   - Confirm correction path
   - Test usefulness

## Examples

### Example 1: Fact Contradiction (Easy)

```json
{
    "document_extract_analysis": "The text discusses Thomas Edison's invention of the light bulb, detailing his use of carbonized bamboo filaments.",
    "testable_concepts": [
        "invention details",
        "material choice",
        "development process"
    ],
    "potential_question_directions": [
        "What specific details about Edison's light bulb invention are highlighted in the text?",
        "In what ways does the text illustrate the significance of the materials used in Edison's invention?",
        "How do the details provided in the text reflect the overall development process of the light bulb?"
    ],
    "best_direction": "What specific details about Edison's light bulb invention are highlighted in the text?", 
    "comprehension_type": "fact_contradiction",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Edison discovered that carbonized bamboo filaments could last over 1200 hours.",
        "This breakthrough came after testing thousands of materials.",
        "The bamboo filaments were far more durable than previous carbon ones."
    ],
    "quote_context": "The quotes establish Edison's use of bamboo filaments.",
    "kind": "false-premise",
    "question": "The text states that Edison's main breakthrough was using steel filaments in his light bulbs. What's wrong with this premise and what does the text actually say?",
    "answer": "The text states Edison used carbonized bamboo filaments, not steel, and this was crucial because they lasted over 1200 hours",
    "reasoning": "Direct contradiction with text's specific mention of bamboo filaments and their importance.",
    "difficulty": 2,
    "difficulty_justification": "Simple factual contradiction easily verified in text."
}
```

### Example 2: Process Inversion (Medium)

```json
{
    "document_extract_analysis": "The passage explains the water cycle, specifically the sequence from evaporation to precipitation.",
    "testable_concepts": [
        "water cycle stages",
        "process sequence",
        "state changes"
    ],
    "potential_question_directions": [
        "What are the stages of the water cycle as described in the text?",
        "How does the passage illustrate the process of evaporation leading to precipitation?",
        "What sequence of events in the water cycle is outlined in the text?"
    ],
    "best_direction": "What are the stages of the water cycle as described in the text?",
    "comprehension_type": "process_inversion",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Water first evaporates from surface water bodies due to solar heating.",
        "The water vapor then rises and condenses into clouds.",
        "Finally, when conditions are right, precipitation occurs."
    ],
    "quote_context": "The quotes establish the correct sequence of the water cycle.",
    "kind": "false-premise",
    "question": "The text suggests that water first forms clouds, then heats up, and finally evaporates from the surface. What's wrong with this premise and what is the actual sequence described?",
    "answer": "The text describes the opposite sequence: water evaporates first due to heating, then rises and forms clouds, before falling as precipitation",
    "reasoning": "The premise inverts the clear sequence presented in the text.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding process sequence and identifying incorrect order."
}
```

### Example 3: System Contradiction (Very Hard)

```json
{
    "document_extract_analysis": "The text explains how the immune system recognizes and responds to specific pathogens.",
    "testable_concepts": [
        "immune recognition",
        "response specificity",
        "system coordination"
    ],
    "potential_question_directions": [
        "What mechanisms enable the immune system to identify and target specific pathogens?",
        "Which elements are crucial for the specificity of immune responses?",
        "In what ways does the immune system coordinate its response to ensure efficiency against pathogens?"
    ],
    "best_direction": "What mechanisms enable the immune system to identify and target specific pathogens?",
    "comprehension_type": "system_contradiction",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Antibodies are produced specifically for each type of pathogen.",
        "Memory cells retain information about previous infections.",
        "This allows for rapid response to repeated exposures.",
        "Different pathogens trigger different antibody types."
    ],
    "quote_context": "The quotes establish the specificity of immune responses.",
    "kind": "false-premise",
    "question": "The text indicates that the immune system produces the same generic antibody for all pathogens and doesn't remember previous infections. What's wrong with these premises and how does the system actually work according to the text?",
    "answer": "The text states that antibodies are pathogen-specific, not generic, and memory cells retain information about previous infections for rapid future response",
    "reasoning": "Multiple contradictions with the text's description of immune system specificity and memory.",
    "difficulty": 5,
    "difficulty_justification": "Requires understanding complex system behavior and identifying multiple contradictions."
}
```

## Common Pitfalls to Avoid

1. **Subtle Contradictions**
   ❌ "The sky was slightly blue instead of very blue."
   ✅ "The text states water freezes at 100°C when it actually says 0°C."

2. **External Knowledge Dependence**
   ❌ "Einstein didn't understand relativity."
   ✅ "The text claims Edison used steel instead of the stated bamboo."

3. **Multiple Errors**
   ❌ "Everything in the paragraph is wrong."
   ✅ "The specific claim X contradicts the text's statement Y."

4. **Unclear Corrections**
   ❌ "This might be wrong somehow."
   ✅ "The text explicitly states the opposite of this premise."

## Output Requirements

1. Generate 3-5 false premise questions per text extract
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure clear contradictions with text
4. Include explicit corrections
5. All premises must be clearly false
6. Questions should test understanding

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Easy/Fact)
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

1. **Premise Selection**
   - Clear contradictions
   - Text-based errors
   - Obvious corrections
   - Important concepts

2. **Error Types**
   - Factual errors
   - Process inversions
   - Relationship mistakes
   - System contradictions
   - Definition errors

3. **Difficulty Progression**
   - Simple facts (Level 1-2)
   - Process errors (Level 3)
   - Complex contradictions (Level 4-5)
   - System errors (Level 5)

4. **Response Requirements**
   - Identify error
   - Provide correction
   - Use text evidence
   - Explain contradiction

5. **Evidence Use**
   - Direct quotes
   - Clear references
   - Explicit corrections
   - Text support