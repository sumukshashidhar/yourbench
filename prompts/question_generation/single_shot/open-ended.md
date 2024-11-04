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

### Example 1: Personal Response (Easy)

```json
{
    "document_extract_analysis": "The text discusses Marie Curie's dedication to science despite facing significant gender discrimination in the early 1900s.",
    "testable_concepts": [
        "perseverance",
        "gender discrimination",
        "scientific dedication"
    ],
    "potential_question_directions": [
        "What elements illustrate Curie's commitment to her scientific work despite societal challenges?",
        "In what ways does the text highlight Curie's resilience in the face of gender bias?",
        "How do Curie's actions reflect her beliefs about the role of women in science?"
    ],
    "best_direction": "What elements illustrate Curie's commitment to her scientific work despite societal challenges?",
    "comprehension_type": "personal_response",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Despite being denied access to regular laboratory space because of her gender, Curie conducted experiments in a converted shed.",
        "She became the first woman to win a Nobel Prize, and remains the only woman to win Nobel Prizes in two sciences.",
        "Curie refused to patent her radium-isolation process, insisting that scientific discovery belonged to everyone."
    ],
    "quote_context": "The quotes show Curie's determination and principles.",
    "kind": "open-ended",
    "question": "How do you think Curie's decision to not patent her discoveries reflects her views about the purpose of scientific research?",
    "answer": "Multiple valid responses focusing on scientific accessibility, public good, and research ethics, supported by Curie's actions described in the text",
    "reasoning": "The text provides evidence of Curie's principles through her actions, allowing exploration of her scientific philosophy.",
    "difficulty": 2,
    "difficulty_justification": "Requires personal engagement with text evidence but allows flexible interpretation."
}
```

### Example 2: Problem Exploration (Medium)

```json
{
    "document_extract_analysis": "The passage discusses the impact of social media on modern communication patterns and relationships.",
    "testable_concepts": [
        "communication changes",
        "social relationships",
        "technology impact"
    ],
    "potential_question_directions": [
        "What specific changes in communication patterns are attributed to social media?",
        "In what ways does the text illustrate the effects of social media on interpersonal relationships?",
        "How do the described impacts of social media reflect broader societal trends in communication?"
    ],
    "best_direction": "What specific changes in communication patterns are attributed to social media?",
    "comprehension_type": "problem_exploration",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Social media has increased the frequency of communication while potentially reducing its depth.",
        "Users report having more connections but fewer close relationships.",
        "The platform's design encourages quick interactions over sustained dialogue."
    ],
    "quote_context": "The quotes establish a tension between communication quantity and quality.",
    "kind": "open-ended",
    "question": "How might social media platforms be redesigned to better support meaningful relationships while maintaining their accessibility?",
    "answer": "Multiple valid approaches addressing platform design, user behavior, and communication quality, based on issues identified in the text",
    "reasoning": "The text identifies specific challenges in current social media design, providing a foundation for exploring solutions.",
    "difficulty": 3,
    "difficulty_justification": "Requires analysis of problems and creative solution development while staying grounded in text evidence."
}
```

### Example 3: Future Implications (Very Hard)

```json
{
    "document_extract_analysis": "The text examines emerging artificial intelligence technologies and their potential impact on employment and skill requirements.",
    "testable_concepts": [
        "technological change",
        "workforce adaptation",
        "skill evolution"
    ],
    "potential_question_directions": [
        "What specific effects does AI have on job roles and responsibilities?",
        "In what ways does the text illustrate the evolving skill sets required in an AI-driven workforce?",
        "How might the trends discussed in the text influence future career paths and educational needs?"
    ],
    "best_direction": "What specific effects does AI have on job roles and responsibilities?",
    "comprehension_type": "future_implications",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "AI is already automating routine cognitive tasks across industries.",
        "Workers are increasingly required to develop skills in areas AI cannot easily replicate.",
        "The pace of technological change is accelerating, requiring continuous learning.",
        "Traditional career paths are being disrupted by technological advancement."
    ],
    "quote_context": "The quotes outline current AI impacts and trends.",
    "kind": "open-ended",
    "question": "Based on the trends described in the text, what might be the most valuable human skills in a workforce increasingly shaped by AI, and how might education systems need to evolve to develop these skills?",
    "answer": "Multiple valid responses exploring skill development, educational adaptation, and human-AI interaction, supported by trends identified in the text",
    "reasoning": "The text establishes current patterns and challenges, providing a foundation for exploring future implications.",
    "difficulty": 5,
    "difficulty_justification": "Requires synthesis of trends, consideration of multiple factors, and complex future projection while maintaining text grounding."
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