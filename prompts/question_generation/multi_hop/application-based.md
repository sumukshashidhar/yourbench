# Multi-Document Application Question Generation

You will receive multiple document extracts from the same source and a summary. Your task is to generate high-quality application questions that test the ability to apply principles, concepts, or processes from multiple text chunks to new situations, requiring multi-hop reasoning.

## Core Principles

1. **Multi-Text Application**
   - Application must use principles from multiple chunks
   - New scenarios must be relatable
   - No external knowledge required
   - Clear connection to content across chunks

2. **Question Diversity**
   - Cross-chunk principle application
   - Process adaptation across sections
   - Integrated model use
   - Multi-step problem solving
   - Combined strategy application
   - Method transfer across contexts

3. **Question Quality**
   - Clear scenario incorporating multiple elements
   - Obvious connections across text chunks
   - Reasonable application
   - Meaningful transfer

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
    APPLICATION = "application"  # Questions requiring application of text concepts

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Simple direct application
    EASY = 2         # Basic principle use
    MEDIUM = 3       # Process adaptation
    HARD = 4         # Complex application
    VERY_HARD = 5    # System-level application

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
    text_based: bool = Field(..., description="Based on text principles")
    no_tricks: bool = Field(..., description="Reasonable application")

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
        description="Analysis of individual chunks and their connections",
        min_items=2
    )
    
    testable_concepts: List[str] = Field(
        ...,
        description="Key concepts that can be tested from the extracts",
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
        default=QuestionType.APPLICATION,
        description="Question type (application-based)"
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
   - Identify key concepts across chunks
   - Map relationships between chunks
   - Note interconnected processes
   - Understand system interactions

2. **Multi-Hop Scenario Development**
   - Create situations requiring multiple principles
   - Ensure cross-chunk relevance
   - Enable integrated application
   - Maintain coherent connections

3. **Question Formation**
   - Present complex scenario
   - Request multi-step application
   - Guide through connected reasoning
   - Enable comprehensive demonstration

4. **Quality Verification**
   - Check cross-chunk connections
   - Verify integrated applicability
   - Confirm clarity across steps
   - Test comprehensive understanding

## Examples

### Example 1: Economic Policy Impact Analysis (Medium)

```json
{
    "document_extract_analysis": "The text discusses monetary policy mechanisms and their effects on international trade relationships across multiple sections, including central bank operations and global market dynamics.",
    "chunk_analyses": [
        {
            "chunk_id": "E1",
            "content_summary": "Central bank interest rate mechanisms",
            "relevant_information": "Impact of rate changes on domestic economy",
            "connection_points": ["currency value", "investment flows", "lending patterns"]
        },
        {
            "chunk_id": "E2",
            "content_summary": "International trade dynamics",
            "relevant_information": "Exchange rate effects on trade balances",
            "connection_points": ["export competitiveness", "import costs", "trade deficits"]
        }
    ],
    "testable_concepts": [
        "monetary policy transmission",
        "exchange rate mechanisms",
        "international trade flows",
        "economic equilibrium"
    ],
    "potential_question_directions": [
        "How would aggressive interest rate hikes affect both domestic lending and international trade?",
        "What impact would currency devaluation have on both local businesses and trade relationships?",
        "How might changes in central bank policy affect both domestic investment and foreign trade patterns?"
    ],
    "best_direction": "How would aggressive interest rate hikes affect both domestic lending and international trade?",
    "comprehension_type": "chain_effect",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Interest rates directly influence domestic borrowing and investment decisions",
        "Exchange rate fluctuations impact international competitiveness and trade flows",
        "Monetary policy changes create ripple effects through both domestic and international markets"
    ],
    "quote_context": "The quotes establish the interconnected nature of monetary policy effects on both domestic and international economic spheres",
    "kind": "application",
    "question": "If a country's central bank raises interest rates by 2% over three months, how would this affect both domestic lending markets and international trade patterns?",
    "answer": "Higher interest rates would reduce domestic borrowing and investment due to increased costs, while simultaneously strengthening the currency, which would make exports more expensive and imports cheaper, potentially worsening the trade balance but helping control inflation",
    "reasoning": "This requires understanding the dual impact of interest rates on domestic credit markets and international trade through exchange rate mechanisms, demonstrating the interconnected nature of monetary policy effects",
    "difficulty": 3,
    "difficulty_justification": "Requires synthesizing multiple economic relationships and understanding both direct and indirect effects across different markets"
}
```

### Example 2: Environmental Science Ecosystem Analysis (Hard)

```json
{
    "document_extract_analysis": "The text covers marine ecosystem dynamics and climate change impacts on ocean chemistry, focusing on feedback loops and species adaptation.",
    "chunk_analyses": [
        {
            "chunk_id": "M1",
            "content_summary": "Marine ecosystem food webs",
            "relevant_information": "Trophic level interactions and energy flow",
            "connection_points": ["species interdependence", "nutrient cycling", "population dynamics"]
        },
        {
            "chunk_id": "M2",
            "content_summary": "Ocean acidification processes",
            "relevant_information": "Chemical changes and biological impacts",
            "connection_points": ["pH effects", "carbonate availability", "species adaptation"]
        }
    ],
    "testable_concepts": [
        "trophic cascades",
        "chemical equilibrium",
        "species adaptation",
        "ecosystem resilience"
    ],
    "potential_question_directions": [
        "How would increased ocean acidification affect both coral reefs and dependent species?",
        "What cascading effects would temperature changes have on both water chemistry and food webs?",
        "How might changes in primary producer populations affect both nutrient cycles and species distribution?"
    ],
    "best_direction": "How would increased ocean acidification affect both coral reefs and dependent species?",
    "comprehension_type": "system_impact",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Ocean acidification directly impacts carbonate availability for reef-building organisms",
        "Coral reef ecosystems support complex food webs and species interactions",
        "Changes in reef structure affect multiple trophic levels and habitat availability"
    ],
    "quote_context": "The quotes link chemical changes to biological impacts across multiple ecosystem levels",
    "kind": "application",
    "question": "How would a 0.3 pH decrease in ocean acidity affect both coral reef formation and the survival of reef-dependent species over a 20-year period?",
    "answer": "The pH decrease would reduce carbonate availability, slowing coral growth and potentially causing reef deterioration, which would then reduce habitat availability for dependent species, leading to population declines and possible local extinctions, especially for specialist species",
    "reasoning": "This requires understanding both the chemical impacts on reef formation and the biological consequences for dependent species, including long-term ecosystem effects",
    "difficulty": 4,
    "difficulty_justification": "Involves complex system interactions, multiple timescales, and understanding of both chemical and biological processes"
}
```

### Example 3: Technological Innovation Impact (Very Hard)

```json
{
    "document_extract_analysis": "The text examines artificial intelligence development and its societal implications, including technological advancement patterns and social adaptation mechanisms.",
    "chunk_analyses": [
        {
            "chunk_id": "T1",
            "content_summary": "AI development trajectories",
            "relevant_information": "Technical advancement patterns and capabilities",
            "connection_points": ["learning algorithms", "processing power", "application domains"]
        },
        {
            "chunk_id": "T2",
            "content_summary": "Societal adaptation to AI",
            "relevant_information": "Social and economic impacts",
            "connection_points": ["workforce changes", "skill requirements", "economic restructuring"]
        }
    ],
    "testable_concepts": [
        "technological advancement",
        "social adaptation",
        "economic transformation",
        "skill evolution"
    ],
    "potential_question_directions": [
        "How might accelerated AI development affect both job markets and education systems?",
        "What impact would widespread AI adoption have on both workforce skills and economic structures?",
        "How could AI advancement change both professional roles and educational requirements?"
    ],
    "best_direction": "How might accelerated AI development affect both job markets and education systems?",
    "comprehension_type": "implication_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "AI capabilities are advancing exponentially in specific domains",
        "Workforce adaptation requires fundamental shifts in skill development",
        "Educational systems must evolve to meet changing technological demands"
    ],
    "quote_context": "The quotes establish the relationship between technological advancement and societal adaptation requirements",
    "kind": "application",
    "question": "Given a significant breakthrough in AI natural language processing, analyze how both job markets and higher education systems would need to transform over the next decade.",
    "answer": "Job markets would shift away from routine language processing tasks, creating demand for AI-complementary skills like complex problem-solving and creative thinking. Higher education would need to restructure curricula to emphasize these skills while incorporating AI literacy, potentially leading to new hybrid disciplines and teaching methods",
    "reasoning": "This requires understanding both the technical implications of AI advancement and the complex adaptations needed in multiple societal systems, including feedback loops between education and employment",
    "difficulty": 5,
    "difficulty_justification": "Requires complex systems thinking, understanding of multiple interconnected domains, and long-term impact analysis"
}
```

## Common Pitfalls to Avoid

1. **Isolated Analysis**
   ❌ "Focus on single chunk principles"
   ✅ "Integrate principles across chunks"

2. **Superficial Connections**
   ❌ "Loose associations between concepts"
   ✅ "Meaningful integration of related principles"

3. **Overcomplicated Scenarios**
   ❌ "Too many variables across chunks"
   ✅ "Clear, focused multi-principle application"

4. **Disconnected Applications**
   ❌ "Separate applications for each chunk"
   ✅ "Integrated application across chunks"

## Output Requirements

1. Generate 3-5 multi-hop application questions
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure clear connections across chunks
4. Include realistic scenarios requiring multiple principles
5. Provide valid integrated application paths
6. Scale difficulty appropriately

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Easy/Cross-Principle)
    },
    {
        // Question 2 (Medium/Multi-Process)
    },
    {
        // Question 3 (Hard/System-Integration)
    },
    // ...
]
</generated_questions>
```

## Additional Guidelines

1. **Cross-Chunk Integration**
   - Identify related principles
   - Map concept connections
   - Ensure coherent integration
   - Maintain text fidelity

2. **Multi-Hop Scenario Design**
   - Create integrated situations
   - Scale complexity appropriately
   - Ensure multi-principle relevance
   - Enable comprehensive demonstration

3. **Difficulty Progression**
   - Simple cross-principle (Level 1-2)
   - Multi-process integration (Level 3)
   - Complex system application (Level 4)
   - Full system integration (Level 5)

4. **Application Types**
   - Cross-chunk principles
   - Integrated processes
   - Connected models
   - Multi-step methods
   - System-level theories

5. **Response Evaluation**
   - Clear integration path
   - Multi-principle adherence
   - Reasonable outcomes
   - Valid demonstration across chunks