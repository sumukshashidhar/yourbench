# Document-Based Open-Ended Question Generation

You will receive a document extract and optionally a summary. Your task is to generate high-quality open-ended questions that encourage exploration, discussion, and multiple valid perspectives based on the provided text extract.

## Core Principles

1. **Text-Anchored Exploration**
   - Questions must be grounded in text content
   - Multiple valid answers possible
   - Evidence-based reasoning required
   - Encourages diverse perspectives

2. **Question Diversity**
   - Discussion starters
   - Problem exploration
   - Creative thinking
   - Personal connection
   - Extended reasoning
   - Alternative viewpoints

3. **Question Quality**
   - Clear focus
   - Encourages elaboration
   - Allows multiple approaches
   - Promotes deep thinking

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

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

1. **Topic Identification**
   - Identify discussion-worthy elements
   - Note controversial aspects
   - Spot connection opportunities
   - Find exploration points

2. **Question Development**
   - Frame for exploration
   - Allow multiple perspectives
   - Enable personal connection
   - Encourage creativity

3. **Response Consideration**
   - Consider possible angles
   - Plan discussion paths
   - Anticipate viewpoints
   - Map exploration areas

4. **Quality Verification**
   - Check text grounding
   - Verify openness
   - Confirm depth potential
   - Test engagement level

## Examples

### Example 1: Historical Analysis (Easy)
```json
{
    "document_extract_analysis": "The text examines the Industrial Revolution's impact on urban development and social structures in 19th century England, highlighting changes in working conditions and city growth.",
    "testable_concepts": [
        "urbanization patterns",
        "social class dynamics",
        "technological advancement",
        "labor conditions"
    ],
    "potential_question_directions": [
        "How did rapid industrialization affect family structures?",
        "What role did technology play in urban development?",
        "How did working conditions influence social movements?",
        "What were the environmental impacts of industrial growth?"
    ],
    "best_direction": "How did rapid industrialization affect family structures?",
    "comprehension_type": "cause_effect",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "What unexpected ways did factory work transform traditional family roles during the Industrial Revolution?",
    "answer": "The transformation included children becoming wage earners, extended families splitting up, and women entering the workforce in unprecedented numbers, challenging Victorian family ideals.",
    "reasoning": "The text provides evidence of how industrial work restructured family dynamics, with specific examples of women and children's changing roles, housing patterns, and economic relationships.",
    "difficulty": 2,
    "difficulty_justification": "Requires basic analysis of cause-effect relationships but draws directly from textual evidence without need for complex synthesis.",
    "supporting_quotes": [
        "Factory work drew women and children into the workforce in unprecedented numbers, with children as young as six working twelve-hour shifts",
        "Traditional family households were forced to split as workers crowded into urban tenements, breaking extended family bonds",
        "The 1833 Factory Act's restrictions on child labor marked a fundamental shift in family economic structures"
    ],
    "quote_context": "These quotes demonstrate the direct impact of industrialization on family structures, showing how economic pressures and new labor patterns fundamentally altered traditional Victorian family arrangements. The quotes specifically highlight the entry of women and children into the workforce and the physical separation of extended families."
}
```

### Example 2: Scientific Process (Medium)
```json
{
    "document_extract_analysis": "The passage details recent breakthroughs in quantum computing, focusing on technical challenges and potential applications.",
    "testable_concepts": [
        "quantum mechanics principles",
        "computational limitations",
        "practical applications",
        "research methodology"
    ],
    "potential_question_directions": [
        "How do quantum computers fundamentally differ from classical computers?",
        "What are the main technical obstacles in quantum computing?",
        "How might quantum computing affect current encryption methods?",
        "What industries might be transformed by quantum computing?"
    ],
    "best_direction": "What are the main technical obstacles in quantum computing?",
    "comprehension_type": "process_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How might solving the decoherence problem in quantum computing lead to unexpected advances in other fields of science?",
    "answer": "Solutions could advance understanding of quantum systems, improve material science, and develop new error correction methods applicable across scientific fields.",
    "reasoning": "The text discusses how quantum decoherence relates to broader scientific principles and how solutions might have wide-ranging implications.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding of technical concepts and ability to connect ideas across scientific domains.",
    "supporting_quotes": [
        "The primary challenge of maintaining quantum coherence beyond microseconds continues to limit practical applications",
        "Recent breakthroughs in error correction have implications for fields ranging from materials science to cryptography",
        "The decoherence problem has led to unexpected insights in quantum measurement theory and fundamental physics"
    ],
    "quote_context": "These quotes establish the central challenge of decoherence in quantum computing while highlighting how research into this problem has yielded broader scientific benefits. They demonstrate the interconnected nature of quantum computing challenges with other scientific domains."
}
```

### Example 3: Environmental Systems (Hard)
```json
{
    "document_extract_analysis": "The text explores coral reef ecosystems and their response to climate change, including adaptation mechanisms and ecosystem services.",
    "testable_concepts": [
        "ecosystem resilience",
        "climate adaptation",
        "biodiversity relationships",
        "environmental feedback loops"
    ],
    "potential_question_directions": [
        "How do coral reefs adapt to environmental stress?",
        "What role do symbiotic relationships play in reef survival?",
        "How might reef ecosystems evolve under continued pressure?",
        "What are the cascading effects of reef degradation?"
    ],
    "best_direction": "How do coral reefs adapt to environmental stress?",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "If coral reefs develop heat resistance through evolutionary adaptation, how might this affect the broader marine ecosystem's balance?",
    "answer": "Changes could impact species relationships, nutrient cycles, and ecosystem services, with complex ripple effects throughout marine food webs.",
    "reasoning": "The text describes interconnected ecosystem relationships and adaptation mechanisms, allowing analysis of systemic changes.",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding of complex systems, multiple variables, and ability to project ecological consequences.",
    "supporting_quotes": [
        "Some coral species have shown remarkable heat tolerance, with genetic analysis revealing adaptive mutations in heat shock proteins",
        "Changes in coral-algal symbiotic relationships have cascading effects through the reef ecosystem, affecting fish populations and nutrient cycling",
        "Recent studies show coral adaptation can alter the competitive balance between species, potentially restructuring entire reef communities"
    ],
    "quote_context": "These quotes provide evidence of coral adaptation mechanisms while highlighting the complex interconnections within reef ecosystems. They demonstrate how individual adaptations can have broader systemic effects throughout marine food webs."
}
```

### Example 4: Economic Policy (Very Hard)
```json
{
    "document_extract_analysis": "The passage examines modern monetary policy tools and their effects on economic stability and wealth distribution.",
    "testable_concepts": [
        "monetary policy mechanisms",
        "economic inequality",
        "financial system stability",
        "policy trade-offs"
    ],
    "potential_question_directions": [
        "How do interest rate changes affect different economic groups?",
        "What are the long-term implications of quantitative easing?",
        "How might digital currencies affect monetary policy?",
        "What role does wealth inequality play in policy effectiveness?"
    ],
    "best_direction": "What are the long-term implications of quantitative easing?",
    "comprehension_type": "implication_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How might the current monetary policy tools need to evolve if global wealth inequality continues to increase while traditional economic indicators show growth?",
    "answer": "Policy evolution could include new distribution mechanisms, alternative measurement metrics, and reformed financial instruments to address systemic inequalities.",
    "reasoning": "The text provides foundation for analyzing policy limitations and potential reforms based on current economic trends and challenges.",
    "difficulty": 5,
    "difficulty_justification": "Requires deep understanding of economic systems, policy mechanisms, and ability to synthesize complex socioeconomic factors.",
    "supporting_quotes": [
        "Traditional economic indicators like GDP growth have shown consistent improvement while wealth inequality metrics reach historic highs",
        "Current monetary policy tools show diminishing effectiveness in addressing distributional challenges",
        "Analysis suggests quantitative easing has disproportionately benefited asset holders while having limited impact on wage growth"
    ],
    "quote_context": "These quotes highlight the growing disconnect between traditional economic measures and wealth distribution, while demonstrating the limitations of current policy tools in addressing systemic inequalities."
}
```

### Example 5: Cultural Analysis (Medium-Hard)
```json
{
    "document_extract_analysis": "The text discusses how digital technology is reshaping cultural traditions and intergenerational knowledge transfer.",
    "testable_concepts": [
        "cultural preservation",
        "technological impact",
        "generational differences",
        "traditional practices"
    ],
    "potential_question_directions": [
        "How does digital documentation affect oral traditions?",
        "What role does technology play in cultural evolution?",
        "How might traditional practices adapt to digital spaces?",
        "What are the benefits and drawbacks of digital preservation?"
    ],
    "best_direction": "How does digital documentation affect oral traditions?",
    "comprehension_type": "relationship_comprehension",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How might the intersection of digital technology and traditional storytelling create new forms of cultural expression that preserve ancient wisdom while engaging younger generations?",
    "answer": "New hybrid forms could emerge, combining traditional narrative structures with interactive digital elements, creating evolving cultural practices.",
    "reasoning": "The text explores the tension between traditional and digital methods of cultural transmission, suggesting possibilities for synthesis.",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding of cultural dynamics, technological impact, and ability to envision innovative solutions while respecting traditions.",
    "supporting_quotes": [
        "Digital documentation of oral traditions has preserved stories but altered traditional transmission patterns",
        "Young people increasingly engage with cultural practices through digital platforms, creating hybrid forms of expression",
        "Elder storytellers have begun incorporating interactive digital elements while maintaining traditional narrative structures"
    ],
    "quote_context": "These quotes demonstrate the evolving relationship between digital technology and traditional cultural practices, showing both the preservation benefits and transformative effects of digital documentation on cultural transmission."
}
```

## Common Pitfalls to Avoid

1. **Closed-Ended Framing**
   ❌ "Is social media good or bad?"
   ✅ "How might social media's impact on relationships evolve?"

2. **Ungrounded Speculation**
   ❌ "What will robots be like in 1000 years?"
   ✅ "Given the AI trends described, what challenges might we face?"

3. **Too Personal**
   ❌ "What's your favorite social media platform?"
   ✅ "How do the communication patterns described affect different types of relationships?"

4. **Overly Broad**
   ❌ "What is the meaning of life?"
   ✅ "How do the ethical principles demonstrated by Curie apply to modern scientific challenges?"

## Output Requirements

1. Generate 3-5 open-ended questions per text extract
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure questions allow multiple valid approaches
4. Include clear text connections
5. Provide thought-provoking exploration paths
6. Balance structure and openness

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1
    },
    {
        // Question 2 
    },
    {
        // Question 3 
    },
    // ...
]
</generated_questions>
```

## Additional Guidelines

1. **Question Framing**
   - Use expansive language
   - Encourage exploration
   - Allow multiple perspectives
   - Enable personal connection

2. **Response Pathways**
   - Consider multiple approaches
   - Plan discussion routes
   - Enable creative thinking
   - Support diverse viewpoints

3. **Difficulty Progression**
   - Personal connection (Level 1-2)
   - Problem exploration (Level 3)
   - Complex consideration (Level 4)
   - Deep investigation (Level 5)

4. **Open-Ended Types**
   - Personal: Individual connection
   - Problem: Issue exploration
   - Creative: Novel approaches
   - Ethical: Moral considerations
   - Future: Forward-looking analysis
   - Alternative: Different viewpoints

5. **Response Evaluation**
   - Multiple valid perspectives
   - Evidence-based reasoning
   - Creative thinking
   - Personal engagement