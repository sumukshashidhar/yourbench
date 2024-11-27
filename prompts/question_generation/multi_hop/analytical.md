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

### Example 1: Historical Event Analysis (Easy)

```json
{
    "document_extract_analysis": "The passages examine the causes and consequences of the Industrial Revolution across different regions and time periods, highlighting technological, social, and economic transformations.",
    "chunk_analyses": [
        {
            "chunk_id": "tech_innovation",
            "content_summary": "Key technological breakthroughs and their spread",
            "relevant_information": "Steam engine development and factory systems",
            "connection_points": ["technological adoption", "economic impact"]
        },
        {
            "chunk_id": "social_change",
            "content_summary": "Social transformations in urban areas",
            "relevant_information": "Urbanization patterns and working conditions",
            "connection_points": ["population movement", "living standards"]
        }
    ],
    "testable_concepts": [
        "technological diffusion",
        "urbanization patterns",
        "social mobility changes",
        "economic transformation"
    ],
    "potential_question_directions": [
        "What drove urban population growth during industrialization?",
        "How did technological adoption affect social structures?",
        "Why did certain regions industrialize faster than others?"
    ],
    "best_direction": "What drove urban population growth during industrialization?",
    "comprehension_type": "cause_effect",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Steam-powered factories created unprecedented demand for urban workers",
        "Rural populations migrated to cities seeking higher wages",
        "New transportation networks enabled mass population movement"
    ],
    "quote_context": "The quotes establish the relationship between technological change and demographic shifts.",
    "kind": "analytical",
    "question": "What factors drove the massive population shift from rural to urban areas during the Industrial Revolution?",
    "answer": "The combination of steam-powered factories creating job opportunities, higher urban wages, and improved transportation networks enabled and encouraged rural populations to migrate to cities",
    "reasoning": "The text shows how technological innovations created economic incentives for urbanization, while infrastructure improvements made such movement possible.",
    "difficulty": 2,
    "difficulty_justification": "Requires connecting basic cause-effect relationships across technological and social changes, but follows a clear logical sequence."
}
```

### Example 2: Scientific Process Integration (Medium)

```json
{
    "document_extract_analysis": "The passages detail the complex process of photosynthesis and cellular respiration, emphasizing the interconnected nature of these biological processes at multiple scales.",
    "chunk_analyses": [
        {
            "chunk_id": "photosynthesis",
            "content_summary": "Light-dependent and independent reactions",
            "relevant_information": "Energy conversion and glucose production",
            "connection_points": ["chemical processes", "energy transfer"]
        },
        {
            "chunk_id": "cell_respiration",
            "content_summary": "Cellular energy extraction methods",
            "relevant_information": "ATP production and electron transport",
            "connection_points": ["energy utilization", "chemical cycles"]
        },
        {
            "chunk_id": "ecosystem_impact",
            "content_summary": "Broader ecological effects",
            "relevant_information": "Carbon cycling and oxygen production",
            "connection_points": ["environmental impact", "global cycles"]
        }
    ],
    "testable_concepts": [
        "energy transformation",
        "chemical cycling",
        "cellular processes",
        "ecosystem interactions"
    ],
    "potential_question_directions": [
        "How do cellular processes contribute to ecosystem balance?",
        "Why is energy transformation efficiency important?",
        "What connects molecular and ecological processes?"
    ],
    "best_direction": "How do cellular processes contribute to ecosystem balance?",
    "comprehension_type": "process_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Photosynthesis converts light energy into chemical energy",
        "Cellular respiration releases stored energy for life processes",
        "These processes drive global carbon and oxygen cycles"
    ],
    "quote_context": "The quotes demonstrate the connection between cellular processes and global ecological cycles.",
    "kind": "analytical",
    "question": "How do the complementary processes of photosynthesis and cellular respiration maintain ecological balance at both cellular and ecosystem levels?",
    "answer": "Photosynthesis and cellular respiration work together to maintain balance through energy conversion and chemical cycling, connecting cellular activities to ecosystem-wide processes of carbon and oxygen exchange",
    "reasoning": "The text illustrates how molecular-level processes scale up to affect global ecological systems through interconnected chemical and energy transformations.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding both cellular processes and their ecological implications, while connecting multiple levels of biological organization."
}
```

### Example 3: Economic Policy Analysis (Hard)

```json
{
    "document_extract_analysis": "The passages examine monetary and fiscal policy interactions during economic crises, focusing on government responses and market outcomes.",
    "chunk_analyses": [
        {
            "chunk_id": "monetary_policy",
            "content_summary": "Central bank interventions and effects",
            "relevant_information": "Interest rate adjustments and market liquidity",
            "connection_points": ["economic stability", "market response"]
        },
        {
            "chunk_id": "fiscal_policy",
            "content_summary": "Government spending and taxation",
            "relevant_information": "Stimulus measures and debt management",
            "connection_points": ["economic growth", "public finance"]
        },
        {
            "chunk_id": "market_outcomes",
            "content_summary": "Economic indicators and market behavior",
            "relevant_information": "Employment and inflation patterns",
            "connection_points": ["policy effectiveness", "economic indicators"]
        }
    ],
    "testable_concepts": [
        "policy coordination",
        "economic stabilization",
        "market dynamics",
        "intervention timing"
    ],
    "potential_question_directions": [
        "How do monetary and fiscal policies interact?",
        "What determines policy effectiveness?",
        "Why do similar interventions produce different outcomes?"
    ],
    "best_direction": "How do monetary and fiscal policies interact?",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Monetary policy affects borrowing costs and investment",
        "Fiscal stimulus influences aggregate demand",
        "Policy coordination determines overall economic impact"
    ],
    "quote_context": "The quotes reveal the complex interactions between different policy tools and their economic effects.",
    "kind": "analytical",
    "question": "Why might aggressive monetary easing be insufficient to stimulate economic recovery without corresponding fiscal policy support?",
    "answer": "Monetary easing alone may fail because low interest rates cannot guarantee spending increases when fiscal policy doesn't support aggregate demand through direct government spending and income support",
    "reasoning": "The text demonstrates how policy effectiveness depends on coordinated action between monetary and fiscal authorities, with each addressing different aspects of economic stability.",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding complex policy interactions, market behavior, and economic theory while synthesizing multiple policy mechanisms."
}
```

### Example 4: Environmental Systems Analysis (Very Hard)

```json
{
    "document_extract_analysis": "The passages explore the interconnections between oceanic thermohaline circulation, atmospheric patterns, and global climate stability.",
    "chunk_analyses": [
        {
            "chunk_id": "ocean_circulation",
            "content_summary": "Deep ocean current patterns",
            "relevant_information": "Salinity and temperature drivers",
            "connection_points": ["heat transport", "global circulation"]
        },
        {
            "chunk_id": "atmospheric_patterns",
            "content_summary": "Wind systems and weather patterns",
            "relevant_information": "Air pressure and precipitation",
            "connection_points": ["climate stability", "weather systems"]
        },
        {
            "chunk_id": "feedback_mechanisms",
            "content_summary": "System interactions and responses",
            "relevant_information": "Regulatory processes and tipping points",
            "connection_points": ["system stability", "climate change"]
        }
    ],
    "testable_concepts": [
        "circulation patterns",
        "feedback loops",
        "system stability",
        "climate regulation"
    ],
    "potential_question_directions": [
        "How do ocean-atmosphere interactions maintain climate stability?",
        "What role do feedback mechanisms play in regulation?",
        "Where are the critical points in the system?"
    ],
    "best_direction": "How do ocean-atmosphere interactions maintain climate stability?",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Thermohaline circulation moves heat between hemispheres",
        "Atmospheric patterns respond to ocean temperature gradients",
        "Multiple feedback loops regulate global climate stability"
    ],
    "quote_context": "The quotes illustrate the complex interactions between oceanic and atmospheric systems in climate regulation.",
    "kind": "analytical",
    "question": "How might a significant disruption to thermohaline circulation trigger cascading effects across global climate systems, and what feedback mechanisms could either amplify or moderate these changes?",
    "answer": "Disruption to thermohaline circulation would alter heat distribution patterns, affecting atmospheric circulation and precipitation patterns, while feedback mechanisms involving salinity, temperature, and wind patterns could either accelerate or dampen these changes through complex system interactions",
    "reasoning": "The text describes interconnected global systems where changes in ocean circulation can trigger multiple feedback loops affecting climate stability through various physical and chemical processes.",
    "difficulty": 5,
    "difficulty_justification": "Requires deep understanding of multiple complex systems, their interactions, and potential feedback mechanisms while considering both direct and indirect effects across global scales."
}
```


### Example 5: Technological Innovation Analysis (Medium-Hard)

```json
{
    "document_extract_analysis": "The passages examine how artificial intelligence development affects various sectors of society, including economic structures, labor markets, and ethical frameworks.",
    "chunk_analyses": [
        {
            "chunk_id": "tech_development",
            "content_summary": "AI advancement patterns and capabilities",
            "relevant_information": "Machine learning breakthroughs and applications",
            "connection_points": ["innovation speed", "technical limitations"]
        },
        {
            "chunk_id": "economic_impact",
            "content_summary": "Labor market transformations",
            "relevant_information": "Job displacement and creation patterns",
            "connection_points": ["workforce changes", "skill demands"]
        },
        {
            "chunk_id": "ethical_implications",
            "content_summary": "Moral and social considerations",
            "relevant_information": "Decision-making frameworks and accountability",
            "connection_points": ["societal impact", "regulatory needs"]
        }
    ],
    "testable_concepts": [
        "technological disruption",
        "labor market evolution",
        "ethical frameworks",
        "regulatory adaptation"
    ],
    "potential_question_directions": [
        "Paradoxes in AI workforce transformation",
        "Competing forces in labor markets",
        "Systemic adaptation challenges"
    ],
    "best_direction": "Paradoxes in AI workforce transformation",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "AI automation primarily affects routine cognitive tasks",
        "New job categories emerge around AI development and maintenance",
        "Skill requirements shift toward human-AI collaboration"
    ],
    "quote_context": "The quotes reveal paradoxical dynamics between automation, job creation, and skill evolution.",
    "kind": "analytical",
    "question": "To what extent does accelerated AI adoption create a self-reinforcing cycle where increased automation simultaneously undermines traditional employment while generating insufficient replacement roles, and could this cycle be interrupted by strategic workforce development or is it an inevitable technological trajectory?",
    "answer": "The evidence suggests a complex feedback loop where AI automation outpaces job creation capacity, but this cycle isn't deterministic - strategic workforce development and human-AI complementary roles can potentially break the pattern, though success depends on institutional adaptation speed and proactive skill development programs",
    "reasoning": "The text demonstrates how technological disruption creates competing forces in labor markets, where the rate of automation and new job creation exist in tension, mediated by institutional capacity for workforce transformation and the emergence of hybrid human-AI roles",
    "difficulty": 4,
    "difficulty_justification": "Requires analyzing multiple interacting systems, understanding feedback loops, and evaluating competing hypotheses about technological determinism versus strategic intervention."
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