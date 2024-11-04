# Document-Based Analytical Question Generation

You will receive a document extract and a summary. Your task is to generate high-quality analytical questions that test deeper understanding, relationships, and implications from the provided text extract.

## Core Principles

1. **Evidence-Based Analysis**
   - Questions must be answerable using the text's content
   - Answers should synthesize information from multiple parts of the text
   - No external knowledge requirements
   - Requires understanding relationships and implications

2. **Question Diversity**
   - Compare and contrast elements
   - Cause and effect relationships
   - Pattern identification
   - Process analysis
   - Evidence evaluation
   - Argument analysis

3. **Question Quality**
   - Clear analytical focus
   - Multiple text elements involved
   - Requires synthesis of information
   - Tests deeper understanding

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

class QuestionType(str, Enum):
    ANALYTICAL = "analytical"  # Questions requiring analysis and synthesis

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Basic comparison or relationship
    EASY = 2         # Simple cause-effect analysis
    MEDIUM = 3       # Multi-factor analysis
    HARD = 4         # Complex relationship analysis
    VERY_HARD = 5    # Deep system/pattern analysis

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
    text_based: bool = Field(..., description="Answerable from text evidence")
    no_tricks: bool = Field(..., description="Avoids misleading complexity")

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
        default=QuestionType.ANALYTICAL,
        description="Question type (analytical)"
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

1. **Relationship Identification**
   - Identify cause-effect relationships
   - Note comparison opportunities
   - Map processes and systems
   - Identify patterns and trends

2. **Evidence Mapping**
   - Locate relevant text passages
   - Connect related information
   - Identify supporting details
   - Map information flow

3. **Question Formation**
   - Focus on relationships and patterns
   - Require information synthesis
   - Target analytical thinking
   - Structure for clarity

4. **Quality Verification**
   - Verify evidence sufficiency
   - Check analytical depth
   - Confirm text-based answering
   - Test for clarity

## Examples

### Example 1: Cause-Effect Analysis (Easy)

```json
{
    "document_extract_analysis": "The passage discusses how increased carbon dioxide levels affect ocean acidification and marine ecosystems.",
    "testable_concepts": [
        "ocean acidification",
        "ecosystem impacts",
        "chemical processes"
    ],
    "potential_question_directions": [
        "In what ways does elevated CO2 contribute to ocean acidification?",
        "What specific effects does ocean acidification have on marine life?",
        "Can you explain the chemical mechanisms behind ocean acidification?"
    ],
    "best_direction": "In what ways does elevated CO2 contribute to ocean acidification?",
    "comprehension_type": "cause_effect",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Rising CO2 levels in the atmosphere cause the ocean to absorb more carbon dioxide.",
        "This absorption leads to increased acidity in ocean waters.",
        "Higher acidity makes it difficult for marine organisms to build calcium carbonate shells."
    ],
    "quote_context": "The quotes establish the causal chain from CO2 to marine impacts.",
    "kind": "analytical",
    "question": "According to the text, how does atmospheric CO2 impact marine organisms' ability to build shells?",
    "answer": "Increased atmospheric CO2 causes ocean acidification, which makes shell formation difficult",
    "reasoning": "The text presents a clear causal chain: increased CO2 → ocean absorption → acidification → difficulty building shells.",
    "difficulty": 2,
    "difficulty_justification": "Requires following a straightforward cause-effect chain through multiple steps."
}
```

### Example 2: Process Analysis (Medium)

```json
{
    "document_extract_analysis": "The text explains the complex process of photosynthesis and its role in plant growth.",
    "testable_concepts": [
        "photosynthesis stages",
        "energy transformation",
        "resource utilization"
    ],
    "potential_question_directions": [
        "What are the key stages involved in the process of photosynthesis?",
        "In what ways does photosynthesis facilitate energy transformation in plants?",
        "How do plants utilize light energy to synthesize glucose during photosynthesis?"
    ],
    "best_direction": "In what ways does photosynthesis facilitate energy transformation in plants?",
    "comprehension_type": "process_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Light energy is captured by chlorophyll in the thylakoid membranes.",
        "This energy drives the conversion of water and CO2 into glucose.",
        "The process requires specific enzymes and occurs in multiple stages."
    ],
    "quote_context": "The quotes outline the key steps and requirements of photosynthesis.",
    "kind": "analytical",
    "question": "Based on the text, explain how plants transform light energy into glucose during photosynthesis.",
    "answer": "Chlorophyll captures light energy, which then drives the conversion of water and CO2 into glucose through multiple enzymatic stages",
    "reasoning": "The text describes the process sequence from light capture to glucose production, including key components.",
    "difficulty": 3,
    "difficulty_justification": "Requires synthesizing multiple process steps and understanding their relationships."
}
```

### Example 3: System Analysis (Very Hard)

```json
{
    "document_extract_analysis": "The passage describes the interconnected feedback loops in climate systems.",
    "testable_concepts": [
        "feedback mechanisms",
        "system interactions",
        "climate stability"
    ],
    "potential_question_directions": [
        "In what ways does the feedback loop of Arctic ice melting contribute to climate change?",
        "What impacts does the reduction of Arctic ice have on global climate patterns?",
        "How does the decrease in Arctic ice influence the overall temperature of the Earth?"
    ],
    "best_direction": "In what ways does the feedback loop of Arctic ice melting contribute to climate change?",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Arctic ice reflects sunlight, helping maintain cooler temperatures.",
        "As ice melts, less sunlight is reflected, leading to warmer temperatures.",
        "Warmer temperatures cause more ice to melt, creating a self-reinforcing cycle.",
        "This positive feedback loop accelerates the warming process."
    ],
    "quote_context": "The quotes describe a complex feedback system in Arctic ice melt.",
    "kind": "analytical",
    "question": "How does the Arctic ice feedback system demonstrate self-reinforcing climate change, according to the text?",
    "answer": "Ice melt reduces sunlight reflection, causing warming that leads to more ice melt, creating an accelerating cycle",
    "reasoning": "The text describes a positive feedback loop where each step amplifies the initial change.",
    "difficulty": 5,
    "difficulty_justification": "Requires understanding complex system interactions and feedback mechanisms."
}
```

## Common Pitfalls to Avoid

1. **Surface-Level Questions**
   ❌ "What is photosynthesis?"
   ✅ "How do the stages of photosynthesis work together to produce glucose?"

2. **Single-Factor Focus**
   ❌ "Does ice reflect sunlight?"
   ✅ "How does ice reflection contribute to climate feedback loops?"

3. **Requiring External Knowledge**
   ❌ "Why is the greenhouse effect important for Earth?"
   ✅ "How do the described feedback mechanisms affect Arctic temperatures?"

4. **Oversimplification**
   ❌ "What happens when ice melts?"
   ✅ "How does ice melt create a self-reinforcing cycle in the climate system?"

## Output Requirements

1. Generate 3-5 analytical questions per text extract
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure questions require synthesis of multiple text elements
4. Include at least one complex system analysis
5. All reasoning must be supported by text evidence
6. Questions should probe relationships and patterns

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Easy/Cause-Effect)
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

1. **Question Formation**
   - Begin with "how" and "why" rather than "what"
   - Focus on relationships between elements
   - Require synthesis of multiple pieces of evidence
   - Target understanding of mechanisms and systems

2. **Evidence Use**
   - Identify multiple relevant quotes
   - Show connections between quotes
   - Support analysis with text evidence
   - Demonstrate relationship patterns

3. **Difficulty Progression**
   - Simple relationships (Level 1-2)
   - Multi-factor analysis (Level 3)
   - Complex systems (Level 4-5)
   - Interconnected patterns (Level 5)

4. **Analysis Types**
   - Compare/Contrast: Similarities and differences
   - Cause/Effect: Chains of causation
   - Process: Steps and mechanisms
   - Systems: Interconnected elements
   - Patterns: Recurring themes or relationships
   - Arguments: Evidence and conclusions

5. **Response Evaluation**
   - Logical connection of elements
   - Support from multiple quotes
   - Clear analytical reasoning
   - Demonstrated understanding of relationships