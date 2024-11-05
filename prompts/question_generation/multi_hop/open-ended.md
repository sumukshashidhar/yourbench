# Multi-Document Open-Ended Question Generation

You will receive multiple document extracts from the same source document and a summary. Your task is to generate high-quality open-ended questions that require synthesizing information across extracts while encouraging exploration, discussion, and multiple valid perspectives.

## Core Principles

1. **Cross-Text Synthesis**
   - Questions must connect multiple text chunks
   - Multiple valid answers possible
   - Evidence-based reasoning required
   - Encourages diverse perspectives

2. **Question Diversity**
   - Multi-hop discussion starters
   - Complex problem exploration
   - Creative synthesis thinking
   - Personal connection with multiple angles
   - Extended cross-chunk reasoning
   - Alternative viewpoints across passages

3. **Question Quality**
   - Clear focus across chunks
   - Encourages multi-faceted elaboration
   - Allows multiple approaches
   - Promotes deep thinking

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
    OPEN_ENDED = "open-ended"  # Questions encouraging exploration and discussion

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Basic personal response
    EASY = 2         # Simple exploration
    MEDIUM = 3       # Thoughtful analysis
    HARD = 4         # Complex consideration
    VERY_HARD = 5    # Deep investigation

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
    text_based: bool = Field(..., description="Grounded in text content")
    no_tricks: bool = Field(..., description="Genuine exploration")

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
        default=QuestionType.OPEN_ENDED,
        description="Question type (open ended)"
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

1. **Cross-Chunk Analysis**
   - Identify connections between chunks
   - Note theme progression
   - Spot synthesis opportunities
   - Find cross-reference points

2. **Multi-Hop Question Development**
   - Frame for cross-chunk exploration
   - Allow multiple synthesis paths
   - Enable comprehensive connection
   - Encourage creative integration

3. **Response Consideration**
   - Consider multiple synthesis angles
   - Plan cross-reference paths
   - Anticipate connection points
   - Map exploration areas

4. **Quality Verification**
   - Check multi-text grounding
   - Verify synthesis requirement
   - Confirm depth potential
   - Test engagement level

## Examples

### Example 1: Cross-Chunk Personal Response (Easy)

```json
{
    "document_extract_analysis": "The text discusses climate change impacts across different regions and time periods.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Arctic ice melt patterns and wildlife impact",
            "relevant_information": "Accelerating ice melt affecting polar bear habitats",
            "connection_points": ["temperature changes", "wildlife adaptation", "ecosystem impact"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Coastal community responses to rising seas",
            "relevant_information": "Communities developing adaptation strategies",
            "connection_points": ["human adaptation", "community response", "environmental change"]
        }
    ],
    "testable_concepts": [
        "environmental adaptation",
        "community resilience",
        "ecosystem change"
    ],
    "potential_question_directions": [
        "How do different groups adapt to environmental changes?",
        "What parallels exist between wildlife and human adaptation?",
        "How do responses to change vary across regions?"
    ],
    "best_direction": "What parallels exist between wildlife and human adaptation?",
    "comprehension_type": "compare_contrast",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Polar bears are changing hunting patterns as ice melts earlier",
        "Coastal communities are developing new building techniques",
        "Both groups show remarkable adaptability to changing conditions"
    ],
    "quote_context": "The quotes demonstrate parallel adaptation strategies across species.",
    "kind": "open-ended",
    "question": "How do the adaptation strategies of wildlife and human communities described in the text reflect different approaches to similar environmental challenges?",
    "answer": "Multiple valid responses comparing adaptation methods, supported by examples from both chunks",
    "reasoning": "The text provides parallel examples of adaptation, allowing exploration of different response strategies.",
    "difficulty": 2,
    "difficulty_justification": "Requires connecting information across chunks but allows flexible interpretation."
}
```

### Example 2: Multi-Hop Problem Exploration (Medium)

```json
{
    "document_extract_analysis": "The text examines technological innovation's impact on workplace culture and productivity across different sectors.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Remote work technology adoption",
            "relevant_information": "Rapid shift to digital collaboration tools",
            "connection_points": ["technology adoption", "workplace change", "communication patterns"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Productivity measurement evolution",
            "relevant_information": "New metrics for remote work effectiveness",
            "connection_points": ["performance metrics", "management adaptation", "work assessment"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Employee well-being initiatives",
            "relevant_information": "Digital wellness programs and support",
            "connection_points": ["employee support", "work-life balance", "mental health"]
        }
    ],
    "testable_concepts": [
        "workplace transformation",
        "digital adaptation",
        "organizational change"
    ],
    "potential_question_directions": [
        "How do different aspects of workplace digitalization interact?",
        "What challenges arise from rapid technological change?",
        "How do organizations balance efficiency and well-being?"
    ],
    "best_direction": "How do organizations balance efficiency and well-being?",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Companies are adopting new collaboration platforms",
        "Traditional productivity metrics are being reconsidered",
        "Digital wellness programs address remote work challenges"
    ],
    "quote_context": "The quotes show the interplay between technology, productivity, and well-being.",
    "kind": "open-ended",
    "question": "How might organizations redesign their approach to workplace technology to better integrate productivity goals with employee well-being, based on the patterns described across the text?",
    "answer": "Multiple valid approaches addressing technology integration, measurement systems, and support structures",
    "reasoning": "The text identifies various aspects of workplace transformation that need integration.",
    "difficulty": 3,
    "difficulty_justification": "Requires synthesis across multiple chunks and consideration of complex interactions."
}
```

### Example 3: Cross-Chunk Future Implications (Very Hard)

```json
{
    "document_extract_analysis": "The text explores space exploration developments across private and public sectors.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Private space company innovations",
            "relevant_information": "New rocket technology and cost reduction",
            "connection_points": ["technology advancement", "commercial space", "innovation"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "International space agency collaboration",
            "relevant_information": "Joint mission planning and resource sharing",
            "connection_points": ["cooperation", "resource allocation", "mission planning"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Space law evolution",
            "relevant_information": "Developing frameworks for space resource use",
            "connection_points": ["regulation", "resource rights", "international law"]
        }
    ],
    "testable_concepts": [
        "space commercialization",
        "international cooperation",
        "regulatory development"
    ],
    "potential_question_directions": [
        "How might private-public space partnerships evolve?",
        "What challenges arise from commercializing space?",
        "How could space law adapt to new developments?"
    ],
    "best_direction": "How might private-public space partnerships evolve?",
    "comprehension_type": "implication_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Private companies are reducing launch costs significantly",
        "Agencies are developing new collaboration frameworks",
        "Space law is adapting to commercial activities"
    ],
    "quote_context": "The quotes establish current trends in space sector development.",
    "kind": "open-ended",
    "question": "Based on the developments described across the text, how might the relationship between private space companies, international agencies, and regulatory frameworks evolve over the next decade, and what challenges might this evolution present?",
    "answer": "Multiple valid responses exploring sector interactions, regulatory adaptation, and development challenges",
    "reasoning": "The text provides foundation for analyzing complex interactions between different space sector elements.",
    "difficulty": 5,
    "difficulty_justification": "Requires synthesis across multiple chunks, consideration of complex interactions, and future projection."
}
```

## Common Pitfalls to Avoid

1. **Single-Chunk Focus**
   ❌ "What does chunk 1 say about Arctic ice?"
   ✅ "How do the described impacts connect across regions?"

2. **Disconnected Synthesis**
   ❌ "Summarize each chunk separately"
   ✅ "How do the patterns in different chunks relate?"

3. **Shallow Integration**
   ❌ "List three facts from each chunk"
   ✅ "How do the concepts evolve across the chunks?"

4. **Missing Connections**
   ❌ "What's interesting about space exploration?"
   ✅ "How do private and public space initiatives interact?"

## Output Requirements

1. Generate 3-5 cross-chunk open-ended questions
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure questions require multiple chunk synthesis
4. Include clear cross-text connections
5. Provide thought-provoking exploration paths
6. Balance structure and openness

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Easy/Personal)
    },
    {
        // Question 2 (Medium/Problem)
    },
    {
        // Question 3 (Hard/Future)
    },
    // ...
]
</generated_questions>
```

## Additional Guidelines

1. **Cross-Chunk Question Framing**
   - Use integrative language
   - Encourage synthesis
   - Allow multiple perspectives
   - Enable comprehensive connection

2. **Multi-Hop Response Pathways**
   - Consider multiple synthesis approaches
   - Plan cross-reference routes
   - Enable creative integration
   - Support diverse viewpoints

3. **Difficulty Progression**
   - Simple synthesis (Level 1-2)
   - Complex integration (Level 3)
   - Multi-factor analysis (Level 4)
   - Deep cross-chunk investigation (Level 5)

4. **Cross-Chunk Types**
   - Comparative: Multiple perspective synthesis
   - Developmental: Progress across chunks
   - Systemic: Interconnected analysis
   - Integrative: Comprehensive connection
   - Predictive: Cross-pattern projection

5. **Response Evaluation**
   - Multiple valid synthesis paths
   - Evidence from multiple chunks
   - Creative integration
   - Comprehensive engagement