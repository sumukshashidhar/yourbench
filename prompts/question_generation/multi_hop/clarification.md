# Multi-Document Clarification Question Generation

You will receive multiple document extracts from the same document and a summary. Your task is to generate high-quality clarification questions that identify areas needing further explanation or understanding across the provided text extracts, focusing on connections and relationships between different chunks.

## Core Principles

1. **Multi-Hop Clarity Focus**
   - Questions target potentially unclear elements across chunks
   - Answers must exist within multiple text chunks
   - Focuses on understanding connections and relationships
   - Identifies cross-chunk ambiguity or complexity

2. **Question Diversity**
   - Term clarification across contexts
   - Process explanation spanning chunks
   - Relationship clarification between sections
   - Context questions linking information
   - Detail explanation across passages
   - Reference resolution between chunks

3. **Question Quality**
   - Multiple focus points across chunks
   - Clear need for cross-chunk explanation
   - Answerable from combined texts
   - Meaningful multi-hop clarification

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

class ChunkAnalysis(BaseModel):
    chunk_id: str
    content_summary: str
    relevant_information: str
    connection_points: List[str]

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
    Represents a structured QA pair for multi-document comprehension testing.
    """
    # Analysis Fields
    document_extract_analysis: str = Field(
        ...,
        description="Analysis of the key points and structure of the extract",
        min_length=50
    )
    
    chunk_analyses: List[ChunkAnalysis] = Field(
        ...,
        min_items=2,
        description="Analysis of individual chunks and their connections"
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

1. **Cross-Chunk Clarity Need Identification**
   - Spot complex terms across chunks
   - Note unclear processes spanning sections
   - Find ambiguous references between chunks
   - Identify assumed knowledge connections

2. **Multi-Hop Clarification Mapping**
   - Locate explanations across chunks
   - Find context clues between sections
   - Track references across passages
   - Check completeness of connected information

3. **Cross-Reference Question Formation**
   - Target specific elements across chunks
   - Frame for clarity in connections
   - Enable precise multi-hop answers
   - Support understanding of relationships

4. **Multi-Hop Quality Verification**
   - Check answer presence across chunks
   - Verify clarity need in connections
   - Confirm cross-reference specificity
   - Test usefulness of relationship understanding

## Examples

### Example 1: Cross-Chunk Term Definition (Easy)

```json
{
    "document_extract_analysis": "The text discusses photosynthesis across multiple sections, with different chunks focusing on structure and process aspects.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Describes cellular structures involved in photosynthesis",
            "relevant_information": "Details about thylakoids and chloroplasts",
            "connection_points": ["Links to energy capture", "Cellular organization"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Explains the light-dependent reactions",
            "relevant_information": "Role of thylakoids in energy capture",
            "connection_points": ["Structure function relationship", "Energy transfer"]
        }
    ],
    "testable_concepts": [
        "photosynthesis components",
        "cellular structures",
        "biological terminology",
        "structure-function relationships"
    ],
    "potential_question_directions": [
        "How do thylakoids' structure support their function in photosynthesis?",
        "What is the relationship between chloroplast organization and energy capture?",
        "How does the arrangement of thylakoids facilitate light-dependent reactions?"
    ],
    "best_direction": "How do thylakoids' structure support their function in photosynthesis?",
    "comprehension_type": "term_definition",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Thylakoids are membrane-bound compartments inside chloroplasts",
        "The stacked arrangement of thylakoids maximizes light capture",
        "Energy transfer occurs across thylakoid membranes during light-dependent reactions"
    ],
    "quote_context": "The quotes connect structural and functional aspects of thylakoids from different sections",
    "kind": "clarification",
    "question": "Based on the information from both passages, how does the structure of thylakoids support their role in photosynthesis?",
    "answer": "Thylakoids' membrane-bound structure and stacked arrangement maximize light capture and facilitate energy transfer during light-dependent reactions",
    "reasoning": "The answer combines structural information from the first chunk with functional details from the second chunk to explain the structure-function relationship",
    "difficulty": 2,
    "difficulty_justification": "Requires connecting structural and functional information across chunks, but relationship is directly stated"
}
```

### Example 2: Cross-Chunk Process Explanation (Medium)

```json
{
    "document_extract_analysis": "The text describes blood clotting across multiple sections, with different chunks covering initiation and cascade processes.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Describes initial platelet response",
            "relevant_information": "Platelet activation and signal release",
            "connection_points": ["Cascade initiation", "Chemical signaling"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Details clotting factor cascade",
            "relevant_information": "Sequential activation of factors",
            "connection_points": ["Signal response", "Clot formation"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Explains final clot structure",
            "relevant_information": "Fibrin mesh formation",
            "connection_points": ["Cascade results", "Final structure"]
        }
    ],
    "testable_concepts": [
        "clotting sequence",
        "signal cascades",
        "cellular interaction",
        "process integration"
    ],
    "potential_question_directions": [
        "How do platelet signals trigger and maintain the clotting cascade?",
        "What is the sequence of events from vessel damage to final clot formation?",
        "How do different clotting factors coordinate throughout the process?"
    ],
    "best_direction": "How do platelet signals trigger and maintain the clotting cascade?",
    "comprehension_type": "process_explanation",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Platelets release chemical signals upon detecting damage",
        "These signals activate the first clotting factors in the cascade",
        "Each activated factor triggers the next in sequence",
        "The cascade culminates in fibrin formation"
    ],
    "quote_context": "The quotes trace the process from initial signaling through cascade completion",
    "kind": "clarification",
    "question": "Could you explain how platelet signals initiate and drive the clotting cascade through to final clot formation?",
    "answer": "Platelets release chemical signals upon detecting damage, which activate initial clotting factors. These factors trigger a sequential cascade, with each factor activating the next, ultimately leading to fibrin formation and clot structure",
    "reasoning": "The answer integrates information across all three chunks to explain the complete process from initiation to completion",
    "difficulty": 3,
    "difficulty_justification": "Requires synthesizing a multi-step process across multiple chunks while maintaining logical sequence"
}
```

### Example 3: Cross-Chunk System-Level Understanding (Very Hard)

```json
{
    "document_extract_analysis": "The text explains immune system coordination across multiple sections, detailing different cell types and their interactions.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Describes B-cell function and antibody production",
            "relevant_information": "B-cell activation and antibody specificity",
            "connection_points": ["T-cell interaction", "Cytokine signaling"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Details T-cell coordination role",
            "relevant_information": "T-cell signaling and response direction",
            "connection_points": ["B-cell activation", "Memory cell development"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Explains memory cell formation",
            "relevant_information": "Long-term immunity development",
            "connection_points": ["Cell type interaction", "Response persistence"]
        }
    ],
    "testable_concepts": [
        "immune coordination",
        "cellular communication",
        "system integration",
        "response development"
    ],
    "potential_question_directions": [
        "How do different immune cells coordinate their responses through cytokine signaling?",
        "What is the relationship between initial response and memory formation?",
        "How do B-cells, T-cells, and memory cells maintain long-term immunity?"
    ],
    "best_direction": "How do different immune cells coordinate their responses through cytokine signaling?",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "B-cells produce antibodies following T-cell signals",
        "T-cells coordinate responses through cytokine release",
        "Memory cells develop from successful B-cell and T-cell interactions",
        "Cytokine networks maintain long-term immune memory"
    ],
    "quote_context": "The quotes demonstrate the interconnected nature of immune cell communication",
    "kind": "clarification",
    "question": "Could you explain how cytokine signaling enables coordination between B-cells, T-cells, and memory cells in developing and maintaining immune responses?",
    "answer": "Cytokines released by T-cells direct B-cells to produce specific antibodies, while also guiding the development of memory cells. This signaling network maintains coordination between all cell types for both immediate response and long-term immunity",
    "reasoning": "The answer synthesizes information about cellular communication and system coordination from all chunks to explain the complex interaction network",
    "difficulty": 5,
    "difficulty_justification": "Requires understanding and connecting multiple levels of cellular interaction and signaling across different text sections while maintaining system-level perspective"
}
```

## Common Pitfalls to Avoid

1. **Single-Chunk Focus**
   ❌ "What are B-cells?"
   ✅ "How do B-cells and T-cells coordinate their responses?"

2. **Missing Connections**
   ❌ "What do T-cells do?"
   ✅ "How do T-cell signals influence B-cell and memory cell development?"

3. **Shallow Integration**
   ❌ "List all immune cells mentioned"
   ✅ "Explain how different immune cells work together through signaling"

4. **Ignoring Relationships**
   ❌ "What is cytokine signaling?"
   ✅ "How does cytokine signaling enable cell-to-cell coordination?"

## Output Requirements

1. Generate 3-5 cross-chunk clarification questions
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure questions target specific clarity needs across chunks
4. Include clear text-based answers drawing from multiple chunks
5. Questions should help resolve cross-chunk ambiguity
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

1. **Cross-Chunk Clarity Need Selection**
   - Technical terms across contexts
   - Complex processes spanning chunks
   - Relationships between sections
   - System interactions across passages
   - Cross-references
   - Connected context needs

2. **Multi-Hop Question Formation**
   - Target multiple chunks
   - Focus on connections
   - Enable integrated answers
   - Support relationship understanding

3. **Cross-Chunk Difficulty Progression**
   - Term relationships (Level 1-2)
   - Process integration (Level 3)
   - System connections (Level 4-5)
   - Complex interactions (Level 5)

4. **Multi-Hop Clarification Types**
   - Terms: Cross-context vocabulary
   - Processes: Multi-step sequences
   - Relationships: Inter-chunk connections
   - Context: Background integration
   - References: Cross-chunk mentions
   - Systems: Complex interactions

5. **Cross-Reference Quality**
   - Clear connection explanation
   - Integrated answers
   - Multi-chunk support
   - Enhanced relationship understanding