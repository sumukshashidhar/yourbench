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

### Example 1: Process Change (Easy)
```json
{
    "document_extract_analysis": "The text details photosynthesis in desert plants, specifically focusing on CAM metabolism adaptations.",
    "testable_concepts": [
        "metabolic adaptations",
        "water conservation",
        "carbon fixation patterns"
    ],
    "potential_question_directions": [
        "How do desert plants modify their photosynthetic process?",
        "What triggers the switch between normal and CAM photosynthesis?",
        "Why is nighttime CO2 absorption advantageous?"
    ],
    "best_direction": "How do desert plants modify their photosynthetic process?",
    "comprehension_type": "process_change",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "What unique metabolic adaptation allows desert plants to perform photosynthesis differently from typical plants?",
    "answer": "Desert plants use CAM metabolism, which allows them to absorb CO2 at night and store it for use during daytime photosynthesis, unlike typical plants that absorb CO2 during the day",
    "reasoning": "This tests understanding of a fundamental process modification that distinguishes desert plants from typical plants, focusing on the temporal shift in CO2 absorption",
    "difficulty": 2,
    "difficulty_justification": "The concept is straightforward and explicitly stated in the text, requiring only basic comprehension of process modification",
    "supporting_quotes": [
        "CAM plants open their stomata at night to collect CO2, storing it as malate in their vacuoles",
        "Unlike typical C3 plants that perform gas exchange during daylight hours, CAM plants have evolved to minimize water loss in arid conditions"
    ],
    "quote_context": "These quotes establish both the unique nighttime CO2 absorption mechanism of CAM plants and explicitly contrast it with normal plant behavior, providing direct evidence for the metabolic adaptation described in the answer",
    "kind": "edge-case"
}
```

### Example 2: Condition Interaction (Medium)
```json
{
    "document_extract_analysis": "The passage explains quantum entanglement breaking under various environmental conditions.",
    "testable_concepts": [
        "quantum coherence",
        "environmental decoherence",
        "measurement effects"
    ],
    "potential_question_directions": [
        "What combination of factors causes entanglement to break?",
        "How does temperature interact with other decoherence factors?",
        "What role does measurement timing play in maintaining entanglement?"
    ],
    "best_direction": "What combination of factors causes entanglement to break?",
    "comprehension_type": "condition_interaction",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How do temperature fluctuations and measurement timing collectively affect quantum entanglement maintenance?",
    "answer": "When temperature exceeds 1K and measurements occur faster than 10^-6 seconds apart, entanglement breaks down due to combined environmental and measurement-induced decoherence",
    "reasoning": "This tests understanding of multiple interacting conditions and their combined effects on a quantum system",
    "difficulty": 3,
    "difficulty_justification": "Requires synthesizing multiple conditions and understanding their interactive effects on a complex quantum phenomenon",
    "supporting_quotes": [
        "Quantum coherence breaks down rapidly when thermal energy exceeds 1 Kelvin",
        "Measurements performed at intervals shorter than 10^-6 seconds introduce significant decoherence effects",
        "The combination of thermal noise and frequent measurements creates an insurmountable barrier to maintaining quantum entanglement"
    ],
    "quote_context": "These quotes establish both the individual thermal and measurement timing thresholds, while the third quote explicitly confirms their combined detrimental effect on entanglement maintenance",
    "kind": "edge-case"
}
```

### Example 3: System Impact (Hard)
```json
{
    "document_extract_analysis": "The text examines cascading failures in power grids during extreme weather events.",
    "testable_concepts": [
        "grid interconnectivity",
        "failure propagation",
        "system resilience"
    ],
    "potential_question_directions": [
        "How do multiple grid failures interact?",
        "What conditions prevent system recovery?",
        "Why do some failures lead to cascading effects while others don't?"
    ],
    "best_direction": "How do multiple grid failures interact?",
    "comprehension_type": "system_impact",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "Under what specific combination of conditions does a local power grid failure transform into a regional blackout?",
    "answer": "A regional blackout occurs when three conditions align: peak demand exceeds 85% capacity, more than two major transmission lines fail, and backup systems can't respond within 30 seconds",
    "reasoning": "Tests understanding of complex system interactions and threshold conditions that lead to catastrophic failures",
    "difficulty": 4,
    "difficulty_justification": "Requires analysis of multiple interacting conditions and understanding of systemic relationships in a complex infrastructure",
    "supporting_quotes": [
        "Grid instability becomes critical when demand surpasses 85% of maximum capacity",
        "The failure of two or more major transmission lines creates unsustainable load redistribution",
        "Backup systems must engage within 30 seconds to prevent cascading failures",
        "When these conditions coincide, localized failures invariably escalate to regional blackouts"
    ],
    "quote_context": "The quotes establish each critical threshold condition and explicitly confirm their combined role in causing regional blackouts, providing direct evidence for the systemic failure conditions",
    "kind": "edge-case"
}
```

### Example 4: Chain Effect (Very Hard)
```json
{
    "document_extract_analysis": "The document describes the interconnected effects of ocean acidification on marine ecosystems.",
    "testable_concepts": [
        "pH threshold effects",
        "species interdependence",
        "ecosystem collapse scenarios"
    ],
    "potential_question_directions": [
        "How do changes in ocean pH trigger cascade effects?",
        "What sequence of events leads to ecosystem collapse?",
        "Why are some species more vulnerable to acidification?"
    ],
    "best_direction": "How do changes in ocean pH trigger cascade effects?",
    "comprehension_type": "chain_effect",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "What sequence of biological and chemical events occurs when ocean pH drops below 7.8, and how does this lead to ecosystem restructuring?",
    "answer": "When pH drops below 7.8, calcifying organisms lose shell integrity within 48 hours, leading to plankton population collapse within 2 weeks, triggering fish population crashes by month 3, and ultimately causing predator species redistribution within 6 months",
    "reasoning": "Tests understanding of complex temporal chains of cause and effect across multiple ecological levels",
    "difficulty": 5,
    "difficulty_justification": "Requires tracking multiple sequential effects across different timescales and understanding complex ecological interactions",
    "supporting_quotes": [
        "At pH levels below 7.8, calcifying organisms show severe shell degradation within 48 hours",
        "Plankton population collapse occurs approximately 14 days after sustained exposure to acidic conditions",
        "Fish populations dependent on planktonic food sources crash within three months of plankton decline",
        "Apex predator redistribution is observed within 6 months of prey population collapse"
    ],
    "quote_context": "These quotes establish the precise timeline and causal chain of events following ocean acidification, providing explicit evidence for each stage of the ecosystem restructuring process",
    "kind": "edge-case"
}
```

### Example 5: Extreme Case (Medium-Hard)
```json
{
    "document_extract_analysis": "The text discusses human adaptation to high-altitude environments.",
    "testable_concepts": [
        "physiological adaptation",
        "oxygen utilization",
        "genetic factors"
    ],
    "potential_question_directions": [
        "What limits human adaptation to extreme altitudes?",
        "How do different adaptation mechanisms interact?",
        "Why do some populations adapt better than others?"
    ],
    "best_direction": "What limits human adaptation to extreme altitudes?",
    "comprehension_type": "extreme_case",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "What physiological changes enable survival at 5000m altitude, and why do these mechanisms fail above 8000m?",
    "answer": "At 5000m, increased red blood cell production and enhanced oxygen utilization enable survival, but above 8000m, these mechanisms become insufficient as oxygen partial pressure drops below 30% of sea level, overwhelming compensatory mechanisms",
    "reasoning": "Tests understanding of physiological adaptations and their limitations under extreme conditions",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding complex physiological mechanisms and their limitations under extreme conditions",
    "supporting_quotes": [
        "Human settlements at 5000m altitude show successful adaptation through increased erythropoiesis and enhanced oxygen utilization efficiency",
        "Above 8000m, where oxygen partial pressure drops below 30% of sea level values, physiological compensation mechanisms become inadequate",
        "No permanent human habitation exists above 8000m due to the fundamental limitations of human physiology"
    ],
    "quote_context": "The quotes directly establish both the successful adaptation mechanisms at 5000m and their failure above 8000m, providing specific evidence for the physiological threshold described",
    "kind": "edge-case"
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