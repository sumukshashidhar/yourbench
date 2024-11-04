# Document-Based Clarification Question Generation

You will receive a document extract and optionally a summary. Your task is to generate high-quality clarification questions that identify areas needing further explanation or understanding in the provided text extract.

## Core Principles

1. **Clarity Focus**
   - Questions target potentially unclear elements
   - Answers must exist within the text
   - Focuses on understanding specifics
   - Identifies ambiguity or complexity

2. **Question Diversity**
   - Term clarification
   - Process explanation
   - Relationship clarification
   - Context questions
   - Detail explanation
   - Reference resolution

3. **Question Quality**
   - Specific focus point
   - Clear need for explanation
   - Answerable from text
   - Meaningful clarification

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

class QuestionType(str, Enum):
    CLARIFICATION = "clarification"  # Questions seeking deeper understanding of specific elements

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Basic term clarification
    EASY = 2         # Simple explanation needs
    MEDIUM = 3       # Process understanding
    HARD = 4         # Complex relationship clarity
    VERY_HARD = 5    # System-level understanding

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
    text_based: bool = Field(..., description="Clarification exists in text")
    no_tricks: bool = Field(..., description="Genuine clarity need")

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
        default=QuestionType.CLARIFICATION,
        description="Question type (clarification)"
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

1. **Clarity Need Identification**
   - Spot complex terms
   - Note unclear processes
   - Find ambiguous references
   - Identify assumed knowledge

2. **Clarification Mapping**
   - Locate explanations
   - Find context clues
   - Track references
   - Check completeness

3. **Question Formation**
   - Target specific elements
   - Frame for clarity
   - Enable precise answers
   - Support understanding

4. **Quality Verification**
   - Check answer presence
   - Verify clarity need
   - Confirm specificity
   - Test usefulness

## Examples

### Example 1: Term Definition (Easy)

```json
{
    "document_extract_analysis": "The text introduces photosynthesis and uses several specialized terms including 'thylakoid' and 'chlorophyll'.",
    "testable_concepts": [
        "photosynthesis components",
        "cellular structures",
        "biological terminology"
    ],
    "potential_question_directions": [
        "What are the essential components that facilitate photosynthesis?",
        "In what ways do thylakoids function within the photosynthesis process?",
        "How do chlorophyll molecules contribute to the overall process of photosynthesis?"
    ],
    "best_direction": "In what ways do thylakoids function within the photosynthesis process?", 
    "comprehension_type": "term_definition",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "The process occurs in thylakoids, membrane-bound compartments inside chloroplasts.",
        "Chlorophyll molecules within the thylakoid capture light energy.",
        "These specialized pigments are essential for photosynthesis."
    ],
    "quote_context": "The text defines thylakoids in relation to photosynthesis.",
    "kind": "clarification",
    "question": "Based on the text's description, what exactly is a thylakoid and what is its role in photosynthesis?",
    "answer": "A thylakoid is a membrane-bound compartment inside chloroplasts where chlorophyll captures light energy",
    "reasoning": "The text provides both the structural definition and functional role of thylakoids.",
    "difficulty": 2,
    "difficulty_justification": "Requires identifying and combining definition and function from text."
}
```

### Example 2: Process Explanation (Medium)

```json
{
    "document_extract_analysis": "The passage describes the complex process of blood clotting, involving multiple steps and factors.",
    "testable_concepts": [
        "clotting sequence",
        "blood components",
        "cellular interaction"
    ],
    "potential_question_directions": [
        "What are the essential stages in the blood clotting process?",
        "In what way do platelets trigger the initiation of blood clotting?",
        "What subsequent actions occur after the activation of platelets?"
    ],
    "best_direction": "In what way do platelets trigger the initiation of blood clotting?",
    "comprehension_type": "process_explanation",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Blood clotting begins when platelets detect vessel damage.",
        "Platelets release chemical signals triggering a cascade of clotting factors.",
        "The cascade results in fibrin formation, creating a mesh that traps blood cells."
    ],
    "quote_context": "The quotes outline the blood clotting sequence.",
    "kind": "clarification",
    "question": "Could you clarify how platelets initiate the blood clotting process and what specific steps follow?",
    "answer": "Platelets detect vessel damage, release chemical signals, which trigger clotting factors, leading to fibrin formation that creates a mesh trapping blood cells",
    "reasoning": "The text describes the sequence of events in the clotting process, from initial detection to final clot formation.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding and clarifying a multi-step process."
}
```

### Example 3: System-Level Understanding (Very Hard)

```json
{
    "document_extract_analysis": "The text explains how different components of the immune system work together to fight infection.",
    "testable_concepts": [
        "immune response",
        "cellular coordination",
        "system integration"
    ],  
    "potential_question_directions": [
        "In what ways do B-cells, T-cells, and memory cells interact through cytokines to effectively respond to infections?",
        "What roles do cytokines play in the communication between different immune cells during an immune response?",
        "How do the interactions among B-cells, T-cells, and memory cells contribute to a coordinated immune response against pathogens?"
    ],
    "best_direction": "In what ways do B-cells, T-cells, and memory cells interact through cytokines to effectively respond to infections?",
    "comprehension_type": "technical_clarification",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "B-cells produce antibodies specific to the detected pathogen.",
        "T-cells coordinate the immune response and directly attack infected cells.",
        "Memory cells retain information about past infections.",
        "These components communicate through chemical signals called cytokines."
    ],
    "quote_context": "The quotes describe immune system component interactions.",
    "kind": "clarification",
    "question": "Could you explain how B-cells, T-cells, and memory cells coordinate their activities through cytokines to mount an effective immune response?",
    "answer": "B-cells produce specific antibodies while T-cells coordinate response and attack infected cells, with both using cytokines for communication, and memory cells storing infection information",
    "reasoning": "The text describes multiple components and their communication methods, requiring clarification of their coordination.",
    "difficulty": 5,
    "difficulty_justification": "Requires clarifying complex interactions between multiple system components."
}
```

## Common Pitfalls to Avoid

1. **Overly Broad Questions**
   ❌ "What is the immune system?"
   ✅ "How do B-cells and T-cells communicate specifically?"

2. **Questions Without Text Support**
   ❌ "What other immune cells exist?"
   ✅ "Could you clarify the role of cytokines mentioned in the text?"

3. **Obvious Questions**
   ❌ "What are cells?"
   ✅ "How do memory cells retain infection information?"

4. **Complex Without Purpose**
   ❌ "What are all possible immune responses?"
   ✅ "Could you explain how the described components work together?"

## Output Requirements

1. Generate 3-5 clarification questions per text extract
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure questions target specific clarity needs
4. Include clear text-based answers
5. Questions should help resolve ambiguity
6. Balance difficulty levels

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Easy/Term)
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

1. **Clarity Need Selection**
   - Technical terms
   - Complex processes
   - Relationships
   - System interactions
   - References
   - Context needs

2. **Question Formation**
   - Be specific
   - Target confusion points
   - Enable clear answers
   - Support understanding

3. **Difficulty Progression**
   - Term clarification (Level 1-2)
   - Process explanation (Level 3)
   - System understanding (Level 4-5)
   - Complex interactions (Level 5)

4. **Clarification Types**
   - Terms: Technical vocabulary
   - Processes: Step sequences
   - Relationships: Connections
   - Context: Background
   - References: Unclear mentions
   - Systems: Complex interactions

5. **Response Quality**
   - Clear explanation
   - Specific answers
   - Text support
   - Enhanced understanding