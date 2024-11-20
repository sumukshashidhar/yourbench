# Document-Based Application Question Generation

You will receive a document extract and optionally a summary. Your task is to generate high-quality application questions that test the ability to apply principles, concepts, or processes from the text to new situations.

## Core Principles

1. **Text-Based Application**
   - Application must use text principles
   - New scenarios must be relatable
   - No external knowledge required
   - Clear connection to text content

2. **Question Diversity**
   - Principle application
   - Process adaptation
   - Model use
   - Problem solving
   - Strategy application
   - Method transfer

3. **Question Quality**
   - Clear scenario
   - Obvious text connection
   - Reasonable application
   - Meaningful transfer

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

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

1. **Principle Identification**
   - Identify key concepts
   - Note processes
   - Map methods
   - Understand systems

2. **Scenario Development**
   - Create relatable situations
   - Ensure principle fit
   - Enable application
   - Maintain relevance

3. **Question Formation**
   - Present clear scenario
   - Request specific application
   - Guide thought process
   - Enable demonstration

4. **Quality Verification**
   - Check text connection
   - Verify applicability
   - Confirm clarity
   - Test usefulness

## Examples

### Example 1: Historical Analysis (Easy)
```json
{
    "document_extract_analysis": "The text discusses the Industrial Revolution's impact on urbanization and social changes in 19th century Britain.",
    "testable_concepts": [
        "urbanization patterns",
        "social mobility",
        "technological impact",
        "labor dynamics"
    ],
    "potential_question_directions": [
        "How might similar industrialization affect a modern rural society?",
        "What social changes would emerge in a developing nation experiencing rapid industrialization?",
        "How would technological advancement impact traditional farming communities today?"
    ],
    "best_direction": "How might similar industrialization affect a modern rural society?",
    "comprehension_type": "pattern_recognition",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How would the industrialization patterns described in the text likely manifest in a present-day agricultural community in a developing country?",
    "answer": "The community would likely experience rapid urbanization, formation of factory-centered settlements, shift from agricultural to industrial labor, and emergence of new social classes, mirroring the British experience but at an accelerated pace due to modern technology.",
    "reasoning": "The text's patterns of industrial development and social change can be mapped onto contemporary situations, accounting for modern factors like advanced technology and global connectivity.",
    "difficulty": 2,
    "difficulty_justification": "Requires straightforward pattern recognition and basic application of historical concepts to a modern context.",
    "supporting_quotes": [
        "The rapid growth of industrial cities led to unprecedented urban migration, with former agricultural workers flooding into manufacturing centers",
        "New social classes emerged as factory owners and skilled workers created distinct societal strata",
        "The transformation of rural communities was complete within a generation, as steam power and mechanization revolutionized production methods"
    ],
    "quote_context": "These quotes demonstrate the key patterns of industrialization: urban migration, social class formation, and technological transformation. They provide direct evidence of the processes that would likely repeat in modern developing nations.",
    "kind": "application"
}
```

### Example 2: Scientific Method Application (Medium)
```json
{
    "document_extract_analysis": "The passage explains quantum entanglement and its implications for particle behavior.",
    "testable_concepts": [
        "quantum correlation",
        "measurement impact",
        "particle interaction",
        "observational effects"
    ],
    "potential_question_directions": [
        "How could entanglement principles improve secure communication?",
        "What implications does quantum correlation have for data transmission?",
        "How might entanglement affect future computing systems?"
    ],
    "best_direction": "How could entanglement principles improve secure communication?",
    "comprehension_type": "concept_application",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "If you were designing a quantum-secure communication system, how would you utilize the entanglement principles described to ensure message privacy?",
    "answer": "You would create entangled particle pairs, distribute them between sender and receiver, and use their correlated states to generate encryption keys, knowing that any interference would be detectable due to quantum measurement effects.",
    "reasoning": "This applies quantum principles to practical security needs while demonstrating understanding of both theoretical concepts and their real-world applications.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding of complex quantum concepts and ability to apply them to practical engineering challenges.",
    "supporting_quotes": [
        "Quantum entanglement creates an instantaneous correlation between particles, regardless of their physical separation",
        "Any attempt to measure or observe an entangled particle immediately affects its paired counterpart",
        "The act of measurement irreversibly changes the quantum state, making interference detection possible"
    ],
    "quote_context": "These quotes establish the fundamental principles of quantum entanglement and measurement effects that would be crucial for secure communication systems, particularly the ability to detect interference.",
    "kind": "application"
}
```

### Example 3: Environmental Systems (Very Hard)
```json
{
    "document_extract_analysis": "The text details coral reef ecosystems and their responses to environmental stressors.",
    "testable_concepts": [
        "ecosystem resilience",
        "environmental adaptation",
        "biodiversity relationships",
        "stress responses"
    ],
    "potential_question_directions": [
        "How would artificial reef systems need to be designed to maximize resilience?",
        "What intervention strategies could protect vulnerable reef species?",
        "How might reef restoration efforts account for multiple stressors?"
    ],
    "best_direction": "How would artificial reef systems need to be designed to maximize resilience?",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "Design a coral reef restoration strategy that incorporates the text's principles about ecosystem resilience while accounting for predicted climate change impacts over the next 50 years.",
    "answer": "The strategy should include: selecting heat-resistant coral species, creating diverse habitat structures, establishing multiple small reef patches for risk distribution, incorporating natural predator-prey relationships, and maintaining water quality parameters within historical ranges.",
    "reasoning": "This requires synthesis of multiple ecosystem principles and their application to a complex, long-term environmental challenge with multiple interacting factors.",
    "difficulty": 5,
    "difficulty_justification": "Demands integration of multiple complex systems, long-term planning, and consideration of numerous interacting variables and future scenarios.",
    "supporting_quotes": [
        "Coral reef resilience depends on multiple factors: species diversity, genetic variation within species, and the presence of functional redundancy in ecosystem roles",
        "Temperature tolerance varies significantly among coral species, with some showing adaptation to higher thermal stress",
        "Reef systems with distributed spatial patterns show greater resistance to environmental stressors"
    ],
    "quote_context": "These quotes outline the key factors in reef resilience and adaptation, providing the scientific basis for designing artificial reef systems that could withstand future environmental challenges.",
    "kind": "application"
}
```

### Example 4: Economic Policy Analysis (Hard)
```json
{
    "document_extract_analysis": "The text examines monetary policy tools and their effects on inflation and economic growth.",
    "testable_concepts": [
        "interest rate mechanisms",
        "inflation control",
        "economic growth factors",
        "policy timing"
    ],
    "potential_question_directions": [
        "How should central banks respond to stagflation?",
        "What policy mix would address both unemployment and inflation?",
        "How might different monetary tools interact during crisis?"
    ],
    "best_direction": "How should central banks respond to stagflation?",
    "comprehension_type": "decision_change",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "Using the monetary policy principles described, how would you address a situation where a country faces both high inflation and economic stagnation?",
    "answer": "Implement a staged approach: first use targeted interest rate adjustments to control inflation while maintaining sector-specific stimulus programs, then gradually shift to broader growth policies once inflation stabilizes, carefully monitoring both metrics to adjust the balance.",
    "reasoning": "This requires balancing competing economic pressures and understanding the complex interactions between different policy tools and their timing.",
    "difficulty": 4,
    "difficulty_justification": "Requires sophisticated understanding of multiple economic principles and their complex interactions in challenging conditions.",
    "supporting_quotes": [
        "Interest rate adjustments have varying lag times for inflation control versus economic growth effects",
        "Sector-specific monetary interventions can provide targeted stimulus while maintaining broader price stability",
        "The effectiveness of monetary policy tools depends heavily on their sequential implementation and timing"
    ],
    "quote_context": "These quotes establish the relationship between different monetary policy tools and their timing effects, which is crucial for addressing the complex challenge of stagflation.",
    "kind": "application"
}
```

### Example 5: Psychological Theory (Medium)
```json
{
    "document_extract_analysis": "The text explores cognitive behavioral therapy principles and their application in treating anxiety disorders.",
    "testable_concepts": [
        "cognitive restructuring",
        "behavioral modification",
        "therapeutic relationship",
        "treatment progression"
    ],
    "potential_question_directions": [
        "How could CBT principles be adapted for group therapy?",
        "What modifications would be needed for online therapy?",
        "How might these techniques work in different cultural contexts?"
    ],
    "best_direction": "How could CBT principles be adapted for group therapy?",
    "comprehension_type": "process_change",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How would you modify the individual CBT techniques described in the text to create an effective group therapy program for social anxiety?",
    "answer": "Adapt individual cognitive restructuring exercises for group discussion, use peer feedback for behavioral experiments, incorporate group dynamics for exposure therapy, and maintain individual progress tracking while leveraging group support mechanisms.",
    "reasoning": "Requires understanding of core CBT principles and ability to adapt them to a different therapeutic context while maintaining their essential elements.",
    "difficulty": 3,
    "difficulty_justification": "Involves adapting established techniques to a new context while maintaining therapeutic effectiveness and managing group dynamics.",
    "supporting_quotes": [
        "Cognitive restructuring involves identifying and challenging maladaptive thought patterns through structured exercises",
        "Behavioral experiments provide real-world validation of modified cognitive frameworks",
        "The therapeutic relationship serves as a safe environment for practicing new coping strategies"
    ],
    "quote_context": "These quotes outline the core components of CBT, demonstrating the key elements that would need to be adapted for a group therapy setting while maintaining therapeutic effectiveness.",
    "kind": "application"
}
```

## Common Pitfalls to Avoid

1. **Unrealistic Scenarios**
   ❌ "Apply ecosystem principles to Mars colonization"
   ✅ "Apply ecosystem principles to a new nature reserve"

2. **External Knowledge Requirement**
   ❌ "Use quantum physics to explain chemistry"
   ✅ "Use the text's principles to analyze a similar situation"

3. **Overly Complex Applications**
   ❌ "Apply to every possible scenario"
   ✅ "Apply to a specific, relevant situation"

4. **Disconnected Scenarios**
   ❌ "Apply economic principles to space travel"
   ✅ "Apply economic principles to a new market situation"

## Output Requirements

1. Generate 3-5 application questions per text extract
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure clear connection to text principles
4. Include realistic scenarios
5. Provide valid application paths
6. Scale difficulty appropriately

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

1. **Application Design**
   - Choose relevant principles
   - Create realistic scenarios
   - Enable clear application
   - Maintain text connection

2. **Scenario Development**
   - Use relatable situations
   - Scale complexity appropriately
   - Ensure principle relevance
   - Enable demonstration

3. **Difficulty Progression**
   - Simple application (Level 1-2)
   - Process adaptation (Level 3)
   - Complex application (Level 4)
   - System application (Level 5)

4. **Application Types**
   - Principles to scenarios
   - Processes to new contexts
   - Models to situations
   - Methods to problems
   - Theories to cases

5. **Response Evaluation**
   - Clear application path
   - Principle adherence
   - Reasonable outcome
   - Valid demonstration