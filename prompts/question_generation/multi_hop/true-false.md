# Multi-Document Comprehension Question Generation

You will receive multiple document extracts from the same source document and a summary. Your task is to generate high-quality true-false questions that test comprehension across these extracts, requiring understanding of relationships and connections between different parts of the text.

## Core Principles

1. **Multi-Document Evidence**
   - Questions MUST be answerable from the provided text extracts
   - Required verbatim quotes from multiple chunks where applicable
   - No external knowledge requirements
   - No inference beyond what's explicitly stated
   - Clear connections between chunks must be established

2. **Question Diversity**
   - Mix of single-chunk and multi-chunk questions
   - Varied difficulty levels (1-5)
   - Different types of comprehension (main ideas, details, relationships)
   - Balance of true and false statements
   - Focus on relationships between information in different chunks

3. **Question Quality**
   - Clear, unambiguous language
   - No trick questions or wordplay
   - Realistic and meaningful assessments
   - Explicit connection points between chunks

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

class ChunkAnalysis(BaseModel):
    chunk_id: str
    content_summary: str
    relevant_information: str
    connection_points: List[str]

class GeneratedQuestionAnswerPair(BaseModel):
    """
    Represents a structured QA pair for multi-document comprehension testing.
    """
    # Analysis Fields
    document_extract_analysis: str = Field(
        ...,
        description="Analysis of the key points and structure of the extract",
        min_length=50
    )
    
    chunk_analyses: List[ChunkAnalysis] = Field(..., min_items=2)
    
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

1. **Multi-Chunk Analysis**
   - Read and understand each chunk carefully
   - Identify connections between chunks
   - Map relationships and dependencies
   - Note potential areas for cross-chunk testing

2. **Concept Identification**
   - List main ideas that span multiple chunks
   - Identify important details that connect chunks
   - Note relationships between concepts across chunks
   - Look for cause-effect relationships that bridge chunks

3. **Question Formation**
   - Create diverse questions requiring multiple chunk comprehension
   - Ensure direct text evidence exists from relevant chunks
   - Vary difficulty levels based on number of chunks needed
   - Maintain balance of true/false answers

4. **Quality Verification**
   - Check each question against quality metrics
   - Verify supporting quotes are verbatim from appropriate chunks
   - Ensure no external knowledge needed
   - Confirm clear, unambiguous language
   - Validate chunk connections

## Examples

Here are diverse examples showing different types of multi-chunk questions:

### Example 1: Cross-Chunk Fact Synthesis (Medium)

```json
{
    "document_extract_analysis": "The extracts discuss climate change impacts, with different chunks covering temperature changes and their effects on ecosystems.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Details global temperature increases over the past century",
            "relevant_information": "1.5°C average temperature rise since 1900",
            "connection_points": ["temperature change", "timeline", "measurement methods"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Describes ecosystem responses to temperature changes",
            "relevant_information": "Species migration patterns shifting northward",
            "connection_points": ["temperature change", "ecosystem impact", "adaptation"]
        }
    ],
    "testable_concepts": [
        "temperature change",
        "ecosystem response",
        "climate impact timeline"
    ],
    "potential_question_directions": [
        "How do temperature changes affect ecosystem behavior?",
        "What is the relationship between global warming and species migration?",
        "How do different aspects of climate change interact?"
    ],
    "best_direction": "How do temperature changes affect ecosystem behavior?",
    "comprehension_type": "relationship",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Global temperatures have risen by 1.5°C since 1900",
        "Species migration patterns have shifted northward in response to warming"
    ],
    "quote_context": "The quotes connect temperature rise to specific ecosystem changes",
    "kind": "true-false",
    "question": "The 1.5°C temperature rise since 1900 has caused species to migrate southward.",
    "answer": "False",
    "reasoning": "While the text confirms a 1.5°C temperature rise, it explicitly states that species migration patterns have shifted northward, not southward, in response to warming.",
    "difficulty": 3,
    "difficulty_justification": "Requires synthesizing information from two chunks and understanding the relationship between temperature change and migration direction."
}
```

### Example 2: Complex System Understanding (Very Hard)

```json
{
    "document_extract_analysis": "The extracts cover different aspects of ocean acidification, its causes, and impacts on marine ecosystems.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Explains CO2 absorption by oceans",
            "relevant_information": "Oceans absorb 25% of atmospheric CO2",
            "connection_points": ["carbon cycle", "ocean chemistry", "atmospheric interaction"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Details pH changes in ocean water",
            "relevant_information": "pH decreased by 0.1 units",
            "connection_points": ["ocean chemistry", "measurement", "timeline"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Describes impact on marine life",
            "relevant_information": "Shell-forming organisms affected by acidification",
            "connection_points": ["ecosystem impact", "species adaptation", "chemical effects"]
        }
    ],
    "testable_concepts": [
        "ocean acidification",
        "carbon absorption",
        "marine ecosystem impact",
        "chemical processes"
    ],
    "potential_question_directions": [
        "How does atmospheric CO2 affect marine life?",
        "What is the relationship between ocean chemistry and ecosystem health?",
        "How do different aspects of ocean acidification interact?"
    ],
    "best_direction": "How do different aspects of ocean acidification interact?",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Oceans absorb 25% of atmospheric CO2",
        "Ocean pH has decreased by 0.1 units",
        "Shell-forming organisms are particularly vulnerable to acidification"
    ],
    "quote_context": "The quotes establish the chain of events from CO2 absorption to biological impacts",
    "kind": "true-false",
    "question": "Ocean acidification affects shell-forming organisms primarily because of increased ocean temperatures rather than pH changes.",
    "answer": "False",
    "reasoning": "The text establishes a clear chain of causation: CO2 absorption leads to pH changes, which directly affect shell-forming organisms. Temperature is not mentioned as the primary factor affecting these organisms.",
    "difficulty": 5,
    "difficulty_justification": "Requires synthesizing information from three chunks and understanding the complex causal chain from CO2 absorption through chemical changes to biological impacts."
}
```

### Example 3: Cross-Chunk Timeline Analysis (Hard)

```json
{
    "document_extract_analysis": "The extracts describe the Industrial Revolution's phases and their environmental impacts over time.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Early Industrial Revolution (1760-1840)",
            "relevant_information": "Coal-based manufacturing begins",
            "connection_points": ["timeline", "technology change", "environmental impact"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Second Industrial Revolution (1870-1914)",
            "relevant_information": "Introduction of electricity and oil usage",
            "connection_points": ["timeline", "energy sources", "technological advancement"]
        }
    ],
    "testable_concepts": [
        "industrial timeline",
        "technological change",
        "energy transition"
    ],
    "potential_question_directions": [
        "How did energy sources change over time?",
        "What was the sequence of industrial development?",
        "How did different phases of industrialization relate?"
    ],
    "best_direction": "How did energy sources change over time?",
    "comprehension_type": "temporal",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Coal-based manufacturing dominated the early Industrial Revolution (1760-1840)",
        "Electricity and oil became prominent energy sources during the Second Industrial Revolution (1870-1914)"
    ],
    "quote_context": "The quotes establish the chronological progression of energy sources",
    "kind": "true-false",
    "question": "Electricity and oil usage preceded coal-based manufacturing in the Industrial Revolution.",
    "answer": "False",
    "reasoning": "The text clearly shows that coal-based manufacturing began in the early Industrial Revolution (1760-1840), while electricity and oil became prominent later, during the Second Industrial Revolution (1870-1914).",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding and comparing chronological information from multiple chunks and recognizing the sequence of technological changes."
}
```

## Common Pitfalls to Avoid

1. **Single-Chunk Focus**
   ❌ Questions that can be answered from one chunk alone
   ✅ Questions that require synthesizing information across chunks

2. **Missing Connections**
   ❌ Questions that don't clearly relate information between chunks
   ✅ Questions that explicitly connect concepts across chunks

3. **Temporal Confusion**
   ❌ Questions that mix up the sequence of events from different chunks
   ✅ Questions that maintain clear chronological relationships

4. **External Knowledge**
   ❌ Questions requiring information beyond the provided chunks
   ✅ Questions answerable solely from the provided chunks

5. **Ambiguous References**
   ❌ Unclear which chunk information comes from
   ✅ Clear attribution and connection of information sources

## Output Requirements

1. Generate 3-5 questions per set of chunks
2. Ensure diverse comprehension types
3. Include at least one question requiring multiple chunk synthesis
4. Include at least one question at difficulty level 4 or 5
5. Provide verbatim supporting quotes from relevant chunks
6. Include complete chunk analyses and quality metrics for each question

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Medium/Cross-Chunk Synthesis)
    },
    {
        // Question 2 (Hard/System Understanding)
    },
    {
        // Question 3 (Very Hard/Timeline Analysis)
    },
    // ...
]
</generated_questions>
```