# Multi-Document Edge Case Question Generation

You will receive multiple document extracts from the same source document and a summary. Your task is to generate high-quality edge case questions that test understanding of boundary conditions, exceptions, and special cases that require synthesizing information across multiple chunks of text.

## Core Principles

1. **Cross-Text Boundaries**
   - Edge cases must span multiple chunks
   - Exceptions must be synthesized
   - Special conditions connected
   - Boundary conditions integrated

2. **Question Diversity**
   - Multi-hop exception cases
   - Cross-chunk boundaries
   - Synthesized circumstances
   - Connected rule limitations
   - Compound extreme cases
   - Inter-related conditions

3. **Question Quality**
   - Clear edge identification across chunks
   - Multi-text supported exceptions
   - Valid synthesized cases
   - Connected boundaries

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
    Represents a structured QA pair for multi-document comprehension testing.
    """
    # Analysis Fields
    document_extract_analysis: str = Field(
        ...,
        description="Analysis of the key points and structure across all extracts",
        min_length=50
    )
    
    chunk_analyses: List[ChunkAnalysis] = Field(
        ...,
        min_items=2,
        description="Analysis of individual chunks and their connections"
    )
    
    testable_concepts: List[str] = Field(
        ...,
        description="Key concepts that can be tested across chunks",
        min_items=2
    )

    potential_question_directions: List[str] = Field(..., description="The possible questions that a human would likely ask")
    best_direction: str = Field(..., description="The best question to ask, a decision made based on the question_directions. Why would it be a good question to ask, and what skills would it test?")

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
        description="Verbatim quotes from multiple chunks that prove the answer",
        min_items=1
    )
    
    quote_context: str = Field(
        ...,
        description="Explanation of how quotes from different chunks support the answer",
        min_length=30
    )

    # Core Question Fields
    kind: QuestionType = Field(
        default=QuestionType.EDGE_CASE,
        description="Question type (edge-case)"
    )
    
    question: str = Field(
        ...,
        description="The multi-hop question"
    )
    
    answer: str = Field(
        ...,
        description="The correct answer synthesizing information across chunks"
    )
    
    reasoning: str = Field(
        ...,
        description="Detailed explanation of the answer showing connections",
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

1. **Cross-Chunk Edge Identification**
   - Spot connected exceptions
   - Find related boundaries
   - Note linked limitations
   - Identify synthesized cases

2. **Multi-Hop Case Development**
   - Define conditions across chunks
   - Map cross-text implications
   - Consider inter-chunk interactions
   - Verify multi-text support

3. **Synthesized Question Formation**
   - Present connected edges
   - Request integrated understanding
   - Enable cross-chunk explanation
   - Guide multi-hop analysis

4. **Quality Verification**
   - Check multi-text basis
   - Verify cross-chunk validity
   - Confirm synthesis clarity
   - Test connection usefulness

## Examples

### Example 1: Cross-Chunk Exception Case (Easy)

```json
{
    "document_extract_analysis": "The texts explain climate adaptation in different species, noting interconnected exceptions across environments.",
    "chunk_analyses": [
        {
            "chunk_id": "A1",
            "content_summary": "Desert species adaptations",
            "relevant_information": "Most desert animals are nocturnal",
            "connection_points": ["temperature adaptation", "behavioral patterns"]
        },
        {
            "chunk_id": "A2",
            "content_summary": "Arctic species behaviors",
            "relevant_information": "Arctic foxes remain active during polar day",
            "connection_points": ["temperature adaptation", "circadian rhythms"]
        }
    ],
    "testable_concepts": [
        "circadian adaptations",
        "environmental exceptions",
        "behavioral patterns"
    ],
    "potential_question_directions": [
        "How do circadian patterns differ between desert and arctic species?",
        "What exceptions exist to typical day/night activity patterns?",
        "How do environmental conditions influence behavioral adaptations?"
    ],
    "best_direction": "What exceptions exist to typical day/night activity patterns?",
    "comprehension_type": "exception_case",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Most desert animals are nocturnal to avoid heat",
        "Arctic foxes maintain activity during constant daylight",
        "These adaptations represent exceptions to normal circadian patterns"
    ],
    "quote_context": "The quotes establish contrasting exceptions to normal activity patterns in different environments",
    "kind": "edge-case",
    "question": "Based on the texts, what specific environmental conditions create exceptions to normal day/night activity patterns in both desert and arctic species?",
    "answer": "Desert animals become nocturnal due to heat, while Arctic foxes remain active during constant daylight, both representing exceptions to typical circadian patterns",
    "reasoning": "The question tests understanding of how different environmental pressures create exceptions to normal behavioral patterns across species",
    "difficulty": 2,
    "difficulty_justification": "Requires connecting simple exceptions from two different chunks but with clear relationship"
}
```

### Example 2: Multi-Hop Boundary Condition (Medium)

```json
{
    "document_extract_analysis": "The passages describe enzyme function across temperature and pH ranges, with interconnected boundary conditions.",
    "chunk_analyses": [
        {
            "chunk_id": "B1",
            "content_summary": "Temperature effects on enzymes",
            "relevant_information": "Enzymes denature above 40°C",
            "connection_points": ["protein stability", "reaction rate"]
        },
        {
            "chunk_id": "B2",
            "content_summary": "pH influence on enzyme activity",
            "relevant_information": "Optimal pH varies by location",
            "connection_points": ["protein stability", "environmental conditions"]
        },
        {
            "chunk_id": "B3",
            "content_summary": "Combined environmental effects",
            "relevant_information": "Temperature and pH interact",
            "connection_points": ["stability conditions", "multiple factors"]
        }
    ],
    "testable_concepts": [
        "enzyme stability",
        "environmental boundaries",
        "condition interactions"
    ],
    "potential_question_directions": [
        "How do temperature and pH interact to affect enzyme function?",
        "What boundary conditions exist for enzyme stability?",
        "When do multiple factors create stability limits?"
    ],
    "best_direction": "How do temperature and pH interact to affect enzyme function?",
    "comprehension_type": "boundary_condition",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Enzymes denature above 40°C",
        "Optimal pH varies by cellular location",
        "Combined effects of temperature and pH can lower stability thresholds"
    ],
    "quote_context": "The quotes establish how multiple factors interact to create boundary conditions",
    "kind": "edge-case",
    "question": "According to the texts, how do the combined effects of temperature and pH create boundary conditions for enzyme stability that differ from their individual effects?",
    "answer": "The texts indicate that while enzymes normally denature above 40°C, the presence of non-optimal pH can lower this temperature threshold, creating a combined boundary condition",
    "reasoning": "Tests understanding of how multiple factors interact to create new boundary conditions beyond their individual effects",
    "difficulty": 3,
    "difficulty_justification": "Requires synthesizing information about multiple interacting factors across three chunks"
}
```

### Example 3: System Boundary Synthesis (Very Hard)

```json
{
    "document_extract_analysis": "The texts discuss ecosystem resilience across different stress conditions, requiring synthesis of multiple interacting factors.",
    "chunk_analyses": [
        {
            "chunk_id": "C1",
            "content_summary": "Species interdependence",
            "relevant_information": "Keystone species effects",
            "connection_points": ["population dynamics", "system stability"]
        },
        {
            "chunk_id": "C2",
            "content_summary": "Resource limitations",
            "relevant_information": "Critical resource thresholds",
            "connection_points": ["carrying capacity", "system stress"]
        },
        {
            "chunk_id": "C3",
            "content_summary": "Recovery patterns",
            "relevant_information": "Tipping points",
            "connection_points": ["system collapse", "resilience factors"]
        },
        {
            "chunk_id": "C4",
            "content_summary": "Compound effects",
            "relevant_information": "Multiple stressor interactions",
            "connection_points": ["synergistic effects", "system boundaries"]
        }
    ],
    "testable_concepts": [
        "ecosystem resilience",
        "tipping points",
        "compound stressors"
    ],
    "potential_question_directions": [
        "How do multiple stressors interact to affect system stability?",
        "What combinations of factors create irreversible change?",
        "When do compound effects exceed system resilience?"
    ],
    "best_direction": "What combinations of factors create irreversible change?",
    "comprehension_type": "system_boundary",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Keystone species loss exceeding 30% reduces stability",
        "Resource depletion below 40% of normal levels stresses systems",
        "Multiple simultaneous stressors can create irreversible tipping points",
        "Recovery becomes impossible when three or more major stressors combine"
    ],
    "quote_context": "The quotes establish multiple interacting conditions that create system boundaries",
    "kind": "edge-case",
    "question": "Based on the texts, what specific combination of conditions creates an irreversible tipping point in ecosystem stability, and how do these factors interact differently than when occurring individually?",
    "answer": "The texts indicate that irreversible change occurs when keystone species loss exceeds 30%, resource levels fall below 40%, and these stressors occur simultaneously, creating effects more severe than the sum of individual impacts",
    "reasoning": "Tests understanding of complex system boundaries created by multiple interacting factors across different aspects of ecosystem function",
    "difficulty": 5,
    "difficulty_justification": "Requires synthesizing multiple complex interactions across four chunks and understanding emergent effects"
}
```

## Common Pitfalls to Avoid

1. **Single-Chunk Focus**
   ❌ "What exception appears in the first passage?"
   ✅ "How do the exceptions described across passages interact?"

2. **Disconnected Analysis**
   ❌ "What are the separate conditions in each text?"
   ✅ "How do the conditions connect across texts?"

3. **Missing Synthesis**
   ❌ "List the boundary conditions from each passage"
   ✅ "How do the boundary conditions interact across passages?"

4. **Isolated Consideration**
   ❌ "What happens in this specific case?"
   ✅ "How do multiple cases combine to create new effects?"

## Output Requirements

1. Generate 3-5 multi-hop edge case questions
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure cases span multiple chunks
4. Include clear cross-chunk connections
5. Provide specific synthesis criteria
6. Scale complexity appropriately

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Easy/Cross-Chunk Exception)
    },
    {
        // Question 2 (Medium/Multi-Hop Boundary)
    },
    {
        // Question 3 (Hard/System Synthesis)
    },
    // ...
]
</generated_questions>
```

## Additional Guidelines

1. **Cross-Chunk Edge Selection**
   - Find connected exceptions
   - Identify related boundaries
   - Note interacting conditions
   - Map system connections

2. **Multi-Hop Development**
   - Define cross-chunk conditions
   - Support with multiple texts
   - Explain interactions
   - Show emergent effects

3. **Difficulty Progression**
   - Simple connections (Level 1-2)
   - Complex interactions (Level 3)
   - Multiple syntheses (Level 4)
   - System interactions (Level 5)

4. **Multi-Hop Edge Types**
   - Connected exceptions
   - Interacting boundaries
   - Compound circumstances
   - System interactions
   - Emergent conditions
   - Synthetic cases

5. **Response Evaluation**
   - Clear connections
   - Multi-text support
   - Valid synthesis
   - Proper interaction