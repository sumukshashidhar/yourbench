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


Markdown
Text
### Example 1

```json
{
    "document_extract_analysis": "The text explores the evolution of artificial intelligence in healthcare, focusing on diagnostic accuracy, patient care transformation, and ethical implications.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "AI diagnostic capabilities and accuracy rates",
            "relevant_information": "Machine learning algorithms showing superior detection rates",
            "connection_points": ["diagnostic accuracy", "technology adoption", "clinical validation"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Patient care personalization through AI",
            "relevant_information": "Customized treatment plans and monitoring",
            "connection_points": ["patient experience", "care optimization", "treatment efficacy"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Ethical considerations in AI healthcare",
            "relevant_information": "Privacy concerns and decision-making accountability",
            "connection_points": ["ethical frameworks", "patient rights", "healthcare access"]
        }
    ],
    "testable_concepts": [
        "AI integration in healthcare",
        "patient-centered care",
        "medical ethics",
        "technological transformation",
        "healthcare equity"
    ],
    "potential_question_directions": [
        "How does AI impact the doctor-patient relationship?",
        "What ethical considerations arise from AI diagnostics?",
        "How might healthcare access change with AI adoption?",
        "What role does patient privacy play in AI healthcare?"
    ],
    "best_direction": "Examining the tension between AI efficiency and human medical care",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "AI algorithms demonstrate 95% accuracy in early disease detection",
        "Personalized treatment plans show 40% better outcomes",
        "Ethical frameworks must balance innovation with patient rights"
    ],
    "quote_context": "The quotes highlight the fundamental tension between technological advancement and maintaining human-centered healthcare delivery.",
    "kind": "open-ended",
    "question": "To what extent can the demonstrated superiority of AI in diagnostic accuracy (95%) and treatment outcomes (40% improvement) justify a reduction in direct physician-patient interaction, given the ethical imperative to maintain human connection in healthcare delivery?",
    "answer": "Multiple valid responses exploring the complex interplay between quantifiable improvements in healthcare outcomes and the intangible benefits of human medical care, considering ethical frameworks and patient rights",
    "reasoning": "The question challenges respondents to weigh concrete performance metrics against abstract human values, while considering systemic implications for healthcare delivery and ethical frameworks",
    "difficulty": 5,
    "difficulty_justification": "Requires sophisticated analysis of quantitative benefits versus qualitative human factors, ethical considerations, and systemic healthcare implications"
}
```

### Example 2

```json
{
    "document_extract_analysis": "The text examines global biodiversity loss across different ecosystems and its cascading effects on climate stability and human societies.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Rainforest ecosystem collapse",
            "relevant_information": "Accelerating species extinction rates",
            "connection_points": ["habitat loss", "species interdependence", "ecosystem stability"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Ocean acidification impacts",
            "relevant_information": "Marine food web disruption",
            "connection_points": ["marine ecosystems", "food security", "climate regulation"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Indigenous knowledge and conservation",
            "relevant_information": "Traditional ecological practices",
            "connection_points": ["cultural preservation", "sustainable practices", "local solutions"]
        }
    ],
    "testable_concepts": [
        "ecosystem interconnectedness",
        "biodiversity preservation",
        "traditional ecological knowledge",
        "environmental justice",
        "conservation strategies"
    ],
    "potential_question_directions": [
        "How do different ecosystem collapses interact?",
        "What role can traditional knowledge play in conservation?",
        "How might biodiversity loss affect human societies?",
        "What are the most effective conservation approaches?"
    ],
    "best_direction": "How do different ecosystem collapses interact?",
    "comprehension_type": "system_impact",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Rainforest species loss accelerating at 1000x natural rate",
        "Ocean acidification threatening 60% of marine species",
        "Indigenous practices maintain 80% of global biodiversity"
    ],
    "quote_context": "The quotes illustrate the interconnected nature of ecosystem collapse and potential solutions.",
    "kind": "open-ended",
    "question": "What cascading effects might emerge from the simultaneous collapse of rainforest and marine ecosystems, and how could traditional ecological knowledge inform potential solutions?",
    "answer": "Multiple valid responses exploring ecosystem interactions, feedback loops, and integration of traditional and modern conservation approaches",
    "reasoning": "The question requires understanding of complex ecological relationships and the value of diverse knowledge systems",
    "difficulty": 5,
    "difficulty_justification": "Requires deep understanding of ecological systems, ability to analyze complex interactions, and integration of different knowledge frameworks"
}
```

### Example 3

```json
{
    "document_extract_analysis": "The text discusses quantum computing developments and their potential impact on cryptography, drug discovery, and financial systems.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Quantum computing breakthroughs",
            "relevant_information": "Recent advances in qubit stability",
            "connection_points": ["technological progress", "computational power", "research milestones"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Cryptographic implications",
            "relevant_information": "Vulnerability of current encryption",
            "connection_points": ["security risks", "data protection", "encryption standards"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Pharmaceutical applications",
            "relevant_information": "Molecular simulation capabilities",
            "connection_points": ["drug discovery", "medical advancement", "research efficiency"]
        }
    ],
    "testable_concepts": [
        "quantum computing applications",
        "cybersecurity evolution",
        "pharmaceutical innovation",
        "technological disruption",
        "risk management"
    ],
    "potential_question_directions": [
        "How might quantum computing transform multiple industries?",
        "What security challenges emerge from quantum advancement?",
        "How could quantum computing accelerate scientific discovery?",
        "What societal impacts might quantum computing create?"
    ],
    "best_direction": "How might quantum computing transform multiple industries?",
    "comprehension_type": "implication_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Quantum computers achieve 100-qubit stability",
        "Current encryption vulnerable to quantum attacks",
        "Molecular simulations 1000x faster with quantum computing"
    ],
    "quote_context": "The quotes demonstrate quantum computing's transformative potential across different sectors.",
    "kind": "open-ended",
    "question": "How might the simultaneous advancement of quantum computing in cryptography and drug discovery reshape our approach to both data security and medical innovation over the next decade?",
    "answer": "Multiple valid responses exploring technological transformation, security adaptation, and scientific advancement",
    "reasoning": "The question requires understanding of quantum computing's diverse applications and ability to project future developments",
    "difficulty": 4,
    "difficulty_justification": "Requires technical understanding, cross-sector analysis, and ability to project future implications"
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