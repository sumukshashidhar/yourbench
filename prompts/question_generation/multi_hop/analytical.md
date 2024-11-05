# Multi-Document Analytical Question Generation

You will receive multiple document extracts from the same source document and a summary. Your task is to generate high-quality analytical questions that test deeper understanding, relationships, and implications across multiple chunks of the provided text extracts.

## Core Principles

1. **Cross-Reference Analysis**
   - Questions must be answerable using content from multiple chunks
   - Answers should synthesize information across different sections
   - No external knowledge requirements
   - Requires understanding relationships and implications between chunks

2. **Question Diversity**
   - Compare and contrast elements across chunks
   - Multi-hop cause and effect relationships
   - Cross-sectional pattern identification
   - Process analysis spanning multiple sections
   - Evidence evaluation across chunks
   - Argument analysis using multiple text segments

3. **Question Quality**
   - Clear analytical focus requiring multiple chunks
   - Multiple text elements involved from different sections
   - Requires synthesis of information across chunks
   - Tests deeper understanding of document-wide concepts

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
    Represents a structured QA pair for multi-chunk document comprehension testing.
    """
    # Analysis Fields
    document_extract_analysis: str = Field(
        ...,
        description="Analysis of the key points and structure across all chunks",
        min_length=50
    )
    
    chunk_analyses: List[ChunkAnalysis] = Field(
        ...,
        description="Analysis of individual chunks and their connections",
        min_items=2
    )
    
    testable_concepts: List[str] = Field(
        ...,
        description="Key concepts that can be tested across chunks",
        min_items=2
    )

    potential_question_directions: List[str] = Field(..., description="The possible questions that a human would likely ask across chunks")
    best_direction: str = Field(..., description="The best multi-hop question to ask, with justification for skills tested")

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
        min_items=2
    )
    
    quote_context: str = Field(
        ...,
        description="Explanation of how quotes from different chunks support the answer",
        min_length=30
    )

    # Core Question Fields
    kind: QuestionType = Field(
        default=QuestionType.ANALYTICAL,
        description="Question type (analytical)"
    )
    
    question: str = Field(
        ...,
        description="The multi-hop question"
    )
    
    answer: str = Field(
        ...,
        description="The correct answer synthesizing multiple chunks"
    )
    
    reasoning: str = Field(
        ...,
        description="Detailed explanation of the answer across chunks",
        min_length=50
    )
    
    difficulty: DifficultyLevel = Field(
        ...,
        description="Question difficulty level"
    )
    
    difficulty_justification: str = Field(
        ...,
        description="Explanation of difficulty rating for multi-hop reasoning",
        min_length=30
    )

    class Config:
        use_enum_values = True
```

## Question Generation Process

1. **Cross-Chunk Relationship Identification**
   - Identify cause-effect relationships spanning chunks
   - Note comparison opportunities across sections
   - Map processes and systems across multiple chunks
   - Identify patterns and trends throughout document

2. **Multi-Hop Evidence Mapping**
   - Locate relevant text passages across chunks
   - Connect related information between sections
   - Identify supporting details from multiple chunks
   - Map information flow across document

3. **Multi-Chunk Question Formation**
   - Focus on relationships and patterns across chunks
   - Require information synthesis from multiple sections
   - Target analytical thinking across document
   - Structure for clarity in multi-hop reasoning

4. **Cross-Reference Quality Verification**
   - Verify evidence sufficiency across chunks
   - Check analytical depth of cross-references
   - Confirm text-based answering using multiple chunks
   - Test for clarity in multi-hop reasoning

## Examples

### Example 1: Cross-Chunk Cause-Effect Analysis (Medium)

```json
{
    "document_extract_analysis": "The passages discuss climate change impacts across different ecosystems, with interconnected effects between Arctic and tropical regions.",
    "chunk_analyses": [
        {
            "chunk_id": "arctic_1",
            "content_summary": "Details Arctic ice melt mechanisms",
            "relevant_information": "Ice melt rates and reflection patterns",
            "connection_points": ["temperature effects", "global circulation"]
        },
        {
            "chunk_id": "tropical_1",
            "content_summary": "Tropical ecosystem responses",
            "relevant_information": "Coral bleaching and marine impacts",
            "connection_points": ["temperature effects", "ocean chemistry"]
        }
    ],
    "testable_concepts": [
        "global climate interconnections",
        "ecosystem feedback loops",
        "temperature impact chains"
    ],
    "potential_question_directions": [
        "How do Arctic changes influence tropical ecosystems?",
        "What connecting mechanisms link polar and equatorial regions?",
        "How do feedback loops connect different climate zones?"
    ],
    "best_direction": "How do Arctic changes influence tropical ecosystems?",
    "comprehension_type": "cause_effect",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Arctic ice melt alters global ocean circulation patterns",
        "Changes in circulation affect tropical water temperatures",
        "Warmer tropical waters lead to increased coral bleaching"
    ],
    "quote_context": "The quotes establish a causal chain from Arctic to tropical regions through ocean circulation.",
    "kind": "analytical",
    "question": "Based on the text, explain how Arctic ice melt contributes to coral bleaching in tropical regions.",
    "answer": "Arctic ice melt alters ocean circulation patterns, which affects tropical water temperatures, leading to increased coral bleaching",
    "reasoning": "The text presents a multi-step causal chain connecting Arctic changes to tropical impacts through global ocean systems.",
    "difficulty": 3,
    "difficulty_justification": "Requires synthesizing information across chunks and understanding global system connections."
}
```

### Example 2: Cross-Chunk Process Analysis (Hard)

```json
{
    "document_extract_analysis": "The passages describe carbon cycle processes across terrestrial and marine environments.",
    "chunk_analyses": [
        {
            "chunk_id": "land_carbon",
            "content_summary": "Terrestrial carbon absorption processes",
            "relevant_information": "Forest and soil carbon storage",
            "connection_points": ["carbon exchange", "atmospheric levels"]
        },
        {
            "chunk_id": "ocean_carbon",
            "content_summary": "Marine carbon sequestration",
            "relevant_information": "Ocean absorption mechanisms",
            "connection_points": ["carbon exchange", "chemical processes"]
        }
    ],
    "testable_concepts": [
        "carbon cycle interactions",
        "ecosystem carbon exchange",
        "sequestration processes"
    ],
    "potential_question_directions": [
        "How do land and ocean carbon processes interact?",
        "What role do different ecosystems play in carbon storage?",
        "How do carbon exchange mechanisms differ between systems?"
    ],
    "best_direction": "How do land and ocean carbon processes interact?",
    "comprehension_type": "process_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Forests absorb carbon through photosynthesis",
        "Oceans exchange carbon through surface interactions",
        "Atmospheric carbon levels influence both systems"
    ],
    "quote_context": "The quotes show interconnected carbon processes between land and ocean systems.",
    "kind": "analytical",
    "question": "How do terrestrial and marine carbon absorption processes complement each other in the global carbon cycle?",
    "answer": "Forests and oceans work together through complementary absorption mechanisms, with atmospheric carbon levels influencing both systems' effectiveness",
    "reasoning": "The text describes how different ecosystems handle carbon absorption through distinct but interconnected processes.",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding complex interactions between multiple environmental systems and processes."
}
```

### Example 3: Cross-Chunk System Analysis (Very Hard)

```json
{
    "document_extract_analysis": "The passages detail global nutrient cycles and their disruption by human activities.",
    "chunk_analyses": [
        {
            "chunk_id": "nutrient_cycle",
            "content_summary": "Natural nutrient cycling processes",
            "relevant_information": "Nitrogen and phosphorus flows",
            "connection_points": ["ecosystem balance", "human impacts"]
        },
        {
            "chunk_id": "human_impact",
            "content_summary": "Anthropogenic disruptions",
            "relevant_information": "Agricultural and industrial effects",
            "connection_points": ["ecosystem balance", "feedback loops"]
        },
        {
            "chunk_id": "ecosystem_response",
            "content_summary": "Environmental adaptations",
            "relevant_information": "System responses to changes",
            "connection_points": ["feedback loops", "stability mechanisms"]
        }
    ],
    "testable_concepts": [
        "nutrient cycle disruption",
        "ecosystem adaptation",
        "system stability"
    ],
    "potential_question_directions": [
        "How do human activities affect nutrient cycles?",
        "What are the ecosystem responses to disruption?",
        "How do feedback mechanisms maintain stability?"
    ],
    "best_direction": "How do human activities affect nutrient cycles?",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Agricultural runoff increases nitrogen levels",
        "Ecosystems show varied adaptation responses",
        "Feedback mechanisms attempt to restore balance"
    ],
    "quote_context": "The quotes demonstrate complex interactions between human activities and natural systems.",
    "kind": "analytical",
    "question": "How do ecosystem feedback mechanisms respond to human-induced nutrient cycle disruptions, and what are their limitations?",
    "answer": "Ecosystems employ various feedback mechanisms to counter nutrient imbalances, but these mechanisms can be overwhelmed by sustained human impacts",
    "reasoning": "The text describes multiple interacting systems and their responses to anthropogenic changes, including both adaptive mechanisms and their limitations.",
    "difficulty": 5,
    "difficulty_justification": "Requires synthesizing complex system interactions across multiple chunks and understanding feedback mechanisms."
}
```

## Common Pitfalls to Avoid

1. **Single-Chunk Questions**
   ❌ "What is the carbon cycle?"
   ✅ "How do terrestrial and marine carbon cycles interact?"

2. **Isolated Analysis**
   ❌ "What happens in the Arctic?"
   ✅ "How do Arctic changes influence global systems?"

3. **Missing Connections**
   ❌ "List the effects of climate change"
   ✅ "How do climate changes in different regions reinforce each other?"

4. **Oversimplified Relationships**
   ❌ "What causes ice to melt?"
   ✅ "How does ice melt contribute to cascading effects across ecosystems?"

## Output Requirements

1. Generate 3-5 analytical questions requiring multiple chunks
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure questions require synthesis across chunks
4. Include at least one complex system analysis
5. All reasoning must be supported by evidence from multiple chunks
6. Questions should probe relationships and patterns across document sections

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Medium/Cross-Chunk Cause-Effect)
    },
    {
        // Question 2 (Hard/Cross-Chunk Process)
    },
    {
        // Question 3 (Very Hard/Cross-Chunk System)
    },
    // ...
]
</generated_questions>
```

## Additional Guidelines

1. **Cross-Chunk Question Formation**
   - Begin with "how" and "why" across chunks
   - Focus on relationships between sections
   - Require synthesis of multiple pieces of evidence
   - Target understanding of cross-document mechanisms

2. **Multi-Hop Evidence Use**
   - Identify relevant quotes from multiple chunks
   - Show connections between different sections
   - Support analysis with cross-referenced evidence
   - Demonstrate relationship patterns across document

3. **Multi-Chunk Difficulty Progression**
   - Simple cross-references (Level 1-2)
   - Multi-chunk analysis (Level 3)
   - Complex cross-document systems (Level 4-5)
   - Interconnected patterns across all chunks (Level 5)

4. **Cross-Document Analysis Types**
   - Compare/Contrast: Similarities and differences across chunks
   - Cause/Effect: Chains of causation spanning sections
   - Process: Steps and mechanisms across document
   - Systems: Interconnected elements from multiple chunks
   - Patterns: Recurring themes or relationships throughout
   - Arguments: Evidence and conclusions from multiple sections

5. **Cross-Reference Response Evaluation**
   - Logical connection of elements across chunks
   - Support from multiple quotes from different sections
   - Clear analytical reasoning across document
   - Demonstrated understanding of cross-chunk relationships