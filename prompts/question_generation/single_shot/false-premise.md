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

### Example 1: Quantitative Analysis (Easy)
```json
{
    "document_extract_analysis": "The text details NASA's Mars Rover mission, specifically discussing the rover's speed capabilities and distance covered on Mars' surface.",
    "testable_concepts": [
        "rover specifications",
        "operational capabilities",
        "mission achievements",
        "technical limitations"
    ],
    "potential_question_directions": [
        "How does the rover's actual speed compare to initial expectations?",
        "What factors influence the rover's movement capabilities?",
        "How do environmental conditions affect the rover's performance?",
        "What technical specifications enable the rover's successful operation?"
    ],
    "best_direction": "The question about speed comparison effectively tests quantitative comprehension while requiring understanding of technical limitations.",
    "kind": "false-premise",
    "comprehension_type": "quantitative",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "The rover travels at 50 kilometers per hour across Mars' surface. What's incorrect about this statement according to the mission data?",
    "answer": "The text specifies that the rover moves at a maximum speed of 0.14 kilometers per hour due to terrain challenges and safety protocols.",
    "reasoning": "This tests understanding of specific numerical data while highlighting the importance of accurate technical specifications in space missions.",
    "difficulty": 1,
    "difficulty_justification": "Simple numerical comparison that requires only basic fact-checking against the text.",
    "supporting_quotes": [
        "The Mars Rover's top speed is limited to 0.14 kilometers per hour",
        "Due to challenging terrain and safety protocols, the rover must move slowly and deliberately across the Martian surface"
    ],
    "quote_context": "These quotes directly contradict the false premise by providing the actual speed limit and explaining why such limitations exist. The first quote gives the specific numerical value, while the second provides contextual reasoning for the speed restriction."
}
```

### Example 2: Process Analysis (Medium)
```json
{
    "document_extract_analysis": "The passage explains photosynthesis, detailing the transformation of sunlight into chemical energy and the role of chlorophyll.",
    "testable_concepts": [
        "energy transformation",
        "chemical processes",
        "cellular mechanisms",
        "biological systems"
    ],
    "potential_question_directions": [
        "How does energy flow through the photosynthetic process?",
        "What role do different cellular components play?",
        "How do environmental factors affect photosynthesis?",
        "What are the key stages of the process?"
    ],
    "best_direction": "The energy flow question effectively tests understanding of complex biological processes while maintaining clear assessment criteria.",
    "kind": "false-premise",
    "comprehension_type": "process_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How accurate is the statement that plants release oxygen first, which then enables them to absorb sunlight and produce glucose?",
    "answer": "This is incorrect. The text explains that plants first absorb sunlight, which enables the splitting of water to produce oxygen as a byproduct, while glucose production occurs simultaneously.",
    "reasoning": "Tests understanding of process sequence and causal relationships in biological systems.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding of multiple steps and their correct sequence in a complex biological process.",
    "supporting_quotes": [
        "The process begins when chlorophyll molecules absorb sunlight energy",
        "This absorbed energy drives the splitting of water molecules, releasing oxygen as a byproduct",
        "Simultaneously, the energy is used to combine CO2 and water into glucose molecules"
    ],
    "quote_context": "These quotes establish the correct sequence of photosynthesis, showing that light absorption initiates the process, followed by simultaneous oxygen release and glucose production, directly contradicting the premise's incorrect sequence."
}
```

### Example 3: System Impact (Very Hard)
```json
{
    "document_extract_analysis": "The text examines global economic systems, focusing on international trade relationships and their effects on national economies.",
    "testable_concepts": [
        "economic interdependence",
        "trade dynamics",
        "market effects",
        "global systems"
    ],
    "potential_question_directions": [
        "The relationship between protectionist policies and domestic prosperity",
        "Interconnected nature of modern economic systems",
        "Cascading effects of trade barriers on global supply chains",
        "Balance between national sovereignty and economic interdependence"
    ],
    "best_direction": "The protectionist policy question effectively tests understanding of complex system interactions while challenging assumptions about economic isolation.",
    "kind": "false-premise",
    "comprehension_type": "system_impact",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "A nation's implementation of protective tariffs represents a purely domestic economic decision that strengthens its market independence and shields it from global economic volatility. Consider the accuracy of this interpretation based on the text's analysis.",
    "answer": "The text explicitly states that tariffs create complex ripple effects throughout the global economy, affecting both the implementing country and its trading partners through reduced trade volume, retaliatory measures, and altered supply chains.",
    "reasoning": "Tests understanding of complex system interactions and multi-layered economic relationships while challenging the misconception of economic isolation in modern global markets.",
    "difficulty": 5,
    "difficulty_justification": "Requires deep understanding of system-wide interactions and ability to trace multiple cause-effect relationships while evaluating complex economic assumptions.",
    "supporting_quotes": [
        "When Country A implements new tariffs, its trading partners typically respond with retaliatory measures",
        "The interconnected nature of global trade means that tariff increases often lead to reduced trade volumes across multiple markets",
        "Studies show that protective tariffs frequently result in higher consumer prices and disrupted supply chains in both the implementing nation and its trading partners"
    ],
    "quote_context": "These quotes demonstrate the complex ripple effects of tariff implementation, showing how trade actions affect multiple parties and lead to various economic consequences, directly contradicting the premise of economic isolation and unilateral benefit."
}
```

### Example 4: Relationship Comprehension (Hard)
```json
{
    "document_extract_analysis": "The passage details the relationship between ocean currents and global climate patterns, emphasizing their interconnected nature.",
    "testable_concepts": [
        "oceanic-atmospheric coupling",
        "climate systems",
        "environmental feedback loops",
        "global patterns"
    ],
    "potential_question_directions": [
        "Misconceptions about the Gulf Stream's role in climate",
        "Common misunderstandings of atmospheric-oceanic relationships",
        "Oversimplification of climate system dependencies",
        "Popular myths about ocean current impacts",
        "Assumptions about climate pattern predictability"
    ],
    "best_direction": "The Gulf Stream misconception effectively challenges common oversimplifications while testing deep systems understanding.",
    "kind": "false-premise",
    "comprehension_type": "relationship_comprehension",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "The Gulf Stream's warming effect on Western Europe must be relatively minor, since atmospheric conditions alone determine regional temperatures. Consider this statement's accuracy based on the passage.",
    "answer": "The text explicitly contradicts this assumption, showing that the Gulf Stream significantly moderates Western Europe's climate through direct heat distribution, while participating in a complex feedback loop with atmospheric conditions.",
    "reasoning": "Tests understanding of complex environmental relationships and challenges a common misconception about the relative importance of oceanic influences on climate.",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding of multiple interconnected systems and their bidirectional relationships, while challenging intuitive but incorrect assumptions.",
    "supporting_quotes": [
        "Ocean currents act as global heat distributors, directly influencing regional climate patterns",
        "Changes in atmospheric temperature and wind patterns can alter ocean current behavior, creating a feedback loop",
        "The Gulf Stream's warm waters significantly moderate the climate of Western Europe, demonstrating the direct relationship between oceanic and atmospheric systems"
    ],
    "quote_context": "These quotes directly contradict the premise by establishing the Gulf Stream's major role in climate moderation and highlighting the interconnected nature of oceanic-atmospheric systems, rather than atmospheric dominance."
}
```

### Example 5: Evidence Synthesis (Medium)
```json
{
    "document_extract_analysis": "The text examines the impact of social media on modern communication patterns and interpersonal relationships.",
    "testable_concepts": [
        "communication patterns",
        "social dynamics",
        "technological impact",
        "behavioral changes"
    ],
    "potential_question_directions": [
        "How has social media changed communication?",
        "What evidence supports behavioral changes?",
        "How do different platforms affect interaction?",
        "What are the key trends in modern communication?"
    ],
    "best_direction": "The communication change question effectively tests ability to synthesize multiple pieces of evidence while maintaining clear assessment criteria.",
    "kind": "false-premise",
    "comprehension_type": "evidence_synthesis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "The passage claims that social media has uniformly decreased the quality of personal relationships while increasing isolation. Is this an accurate summary?",
    "answer": "No, the text presents a nuanced view, showing both positive and negative impacts: while some relationships become superficial, others are strengthened through increased connectivity and new forms of interaction.",
    "reasoning": "Tests ability to synthesize multiple pieces of evidence and recognize complexity in social phenomena.",
    "difficulty": 3,
    "difficulty_justification": "Requires synthesis of multiple pieces of evidence and recognition of nuanced conclusions rather than binary outcomes.",
    "supporting_quotes": [
        "Social media platforms have enabled maintenance of long-distance relationships that would have been difficult in previous eras",
        "While some users report increased feelings of isolation, others have found meaningful connections through online communities",
        "Research indicates that the impact of social media on relationships varies significantly based on usage patterns and individual circumstances"
    ],
    "quote_context": "These quotes demonstrate the complex and nuanced effects of social media on relationships, showing both positive and negative impacts rather than the uniform negative effect suggested in the false premise."
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