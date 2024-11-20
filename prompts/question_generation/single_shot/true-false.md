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

### Example 1: Process Analysis (Easy)

```json
{
    "document_extract_analysis": "The text details photosynthesis in plants, explaining how sunlight is converted into chemical energy through a series of biochemical reactions.",
    "testable_concepts": [
        "light absorption",
        "energy conversion",
        "chlorophyll function",
        "glucose production"
    ],
    "potential_question_directions": [
        "How do plants capture and utilize sunlight?",
        "What role does chlorophyll play in photosynthesis?",
        "What is the primary output of photosynthesis?",
        "How does energy transformation occur during photosynthesis?"
    ],
    "best_direction": "What role does chlorophyll play in photosynthesis? This tests understanding of a key component's function in the process.",
    "comprehension_type": "process_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "kind": "true-false",
    "question": "Chlorophyll molecules in plant cells are responsible for capturing sunlight energy during photosynthesis.",
    "answer": "True",
    "reasoning": "The text explicitly states that chlorophyll molecules are the primary light-capturing pigments in plant cells, essential for initiating the photosynthetic process.",
    "difficulty": 2,
    "difficulty_justification": "While it tests process understanding, the relationship between chlorophyll and light capture is clearly stated and represents a fundamental concept.",
    "supporting_quotes": [
        "Chlorophyll molecules embedded in the thylakoid membranes capture incoming solar radiation",
        "These specialized pigment molecules are the primary light-harvesting components of photosynthesis"
    ],
    "quote_context": "These quotes directly establish chlorophyll's role in capturing light energy, confirming the statement's accuracy about chlorophyll's function in photosynthesis"
}
```

### Example 2: System Impact (Very Hard)

```json
{
    "document_extract_analysis": "The passage examines the complex interplay between ocean currents, atmospheric patterns, and global climate systems.",
    "testable_concepts": [
        "thermohaline circulation",
        "atmospheric feedback loops",
        "climate system interactions",
        "heat distribution patterns"
    ],
    "potential_question_directions": [
        "How do changes in ocean currents affect global weather patterns?",
        "What role does salinity play in ocean circulation?",
        "How do atmospheric and oceanic systems interact?",
        "What are the long-term implications of current pattern changes?"
    ],
    "best_direction": "How do changes in ocean currents affect global weather patterns? This tests complex system understanding and interconnected effects.",
    "comprehension_type": "system_impact",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "kind": "true-false",
    "question": "A slowdown in thermohaline circulation would only affect local ocean temperatures without impacting global atmospheric patterns.",
    "answer": "False",
    "reasoning": "The text demonstrates that thermohaline circulation is integrally connected to global atmospheric patterns, and changes would have far-reaching effects beyond local ocean temperatures.",
    "difficulty": 5,
    "difficulty_justification": "Requires understanding of complex system interactions, feedback loops, and the ability to trace multiple cause-effect relationships through different Earth systems.",
    "supporting_quotes": [
        "The thermohaline circulation acts as Earth's heat distribution system, moving warm water from the equator to the poles",
        "Changes in ocean circulation patterns trigger cascading effects throughout the global climate system, altering precipitation patterns, wind systems, and temperature distributions across continents",
        "The interconnected nature of oceanic and atmospheric systems means that perturbations in one component inevitably affect the other"
    ],
    "quote_context": "These quotes demonstrate the interconnected nature of ocean currents and global climate, showing how thermohaline circulation affects both oceanic and atmospheric systems globally"
}
```

### Example 3: Comparative Analysis (Medium)

```json
{
    "document_extract_analysis": "The text compares different types of renewable energy sources, focusing on their efficiency, cost, and environmental impact.",
    "testable_concepts": [
        "energy efficiency metrics",
        "environmental impact factors",
        "cost-benefit analysis",
        "implementation challenges"
    ],
    "potential_question_directions": [
        "What distinguishes solar from wind energy efficiency?",
        "How do environmental impacts compare between different renewables?",
        "What factors influence the cost-effectiveness of each source?",
        "Which renewable source has the lowest environmental impact?"
    ],
    "best_direction": "How do environmental impacts compare between different renewables? This tests ability to analyze and compare multiple factors.",
    "comprehension_type": "compare_contrast",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "kind": "true-false",
    "question": "Solar panels have a higher manufacturing environmental impact but lower operational impact compared to wind turbines.",
    "answer": "True",
    "reasoning": "The text explicitly compares the environmental impacts of different renewable sources, noting this specific distinction between solar and wind power.",
    "difficulty": 3,
    "difficulty_justification": "Requires synthesizing information about two different technologies and understanding both manufacturing and operational impacts.",
    "supporting_quotes": [
        "Solar panel production requires energy-intensive manufacturing processes and rare earth elements, resulting in significant initial environmental impact",
        "While wind turbines have lower manufacturing impacts, their operational phase involves wildlife disruption and noise pollution",
        "Once installed, solar panels operate with minimal environmental impact, producing no emissions or noise pollution"
    ],
    "quote_context": "These quotes explicitly compare the environmental impacts of solar and wind power at different stages of their lifecycles, supporting the statement about their relative impacts"
}
```

### Example 4: Mechanism Understanding (Hard)

```json
{
    "document_extract_analysis": "The passage explains the molecular mechanisms of antibiotic resistance, including genetic transfer and evolutionary adaptation.",
    "testable_concepts": [
        "horizontal gene transfer",
        "mutation mechanisms",
        "selective pressure",
        "resistance development"
    ],
    "potential_question_directions": [
        "How do bacteria develop antibiotic resistance?",
        "What role does genetic transfer play in resistance?",
        "How does selective pressure influence bacterial evolution?",
        "What mechanisms enable rapid adaptation?"
    ],
    "best_direction": "How do bacteria develop antibiotic resistance? This tests understanding of complex biological mechanisms and their interactions.",
    "comprehension_type": "mechanism_understanding",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "kind": "true-false",
    "question": "Horizontal gene transfer allows bacteria to acquire resistance genes only from their direct ancestors.",
    "answer": "False",
    "reasoning": "The text explains that horizontal gene transfer occurs between different bacterial species, not just from ancestors to descendants.",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding of complex biological mechanisms, genetic concepts, and the ability to distinguish between vertical and horizontal gene transfer.",
    "supporting_quotes": [
        "Horizontal gene transfer allows bacteria to share resistance genes across different species and genera",
        "Unlike vertical inheritance from parent to offspring, horizontal gene transfer enables the rapid spread of resistance genes throughout bacterial populations regardless of ancestral relationships",
        "This mechanism of genetic exchange has been observed between entirely unrelated bacterial species in clinical settings"
    ],
    "quote_context": "These quotes directly contradict the statement by showing that horizontal gene transfer occurs between different species, not just from ancestors"
}
```

### Example 5: Evidence Synthesis (Medium)

```json
{
    "document_extract_analysis": "The text discusses archaeological evidence for early human migrations, combining genetic, archaeological, and geological data.",
    "testable_concepts": [
        "migration patterns",
        "dating techniques",
        "archaeological evidence",
        "genetic markers"
    ],
    "potential_question_directions": [
        "How do different types of evidence support migration theories?",
        "What role does genetic evidence play in tracking migrations?",
        "How do archaeologists date ancient human settlements?",
        "What can artifacts tell us about migration routes?"
    ],
    "best_direction": "How do different types of evidence support migration theories? This tests ability to synthesize multiple lines of evidence.",
    "comprehension_type": "evidence_synthesis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "kind": "true-false",
    "question": "Genetic evidence alone provides a complete picture of early human migration patterns.",
    "answer": "False",
    "reasoning": "The text emphasizes that understanding early human migrations requires combining multiple types of evidence, including genetic, archaeological, and geological data.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding how different types of evidence contribute to historical conclusions and why no single source is sufficient.",
    "supporting_quotes": [
        "While genetic evidence provides crucial insights into human migration patterns, it must be corroborated with archaeological findings and geological data",
        "The most robust conclusions about early human migrations come from the synthesis of multiple lines of evidence, including artifact distributions, genetic markers, and dated geological features",
        "No single type of evidence can provide a complete picture of ancient human movements"
    ],
    "quote_context": "These quotes explicitly state that multiple types of evidence are necessary and that genetic evidence alone is insufficient, directly contradicting the statement"
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