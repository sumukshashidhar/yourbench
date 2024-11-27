# Document-Based Factual Question Generation

You will receive a document extract and optionally a summary. Your task is to generate high-quality factual questions that test recall and comprehension of specific information from the provided text extract.

## Core Principles

1. **Direct Text Evidence**
   - Questions MUST be answerable with specific words/phrases from the text
   - Answers must be directly quoted from the text
   - No external knowledge requirements
   - No inference beyond explicit facts

2. **Question Diversity**
   - Mix of entity identification (people, places, things)
   - Temporal questions (when, what year, etc.)
   - Quantitative facts (numbers, amounts, etc.)
   - Relationship identification (who did what, what belongs to whom)

3. **Question Quality**
   - Clear, specific questions with unambiguous answers
   - Answers must be directly present in text
   - Questions should target meaningful facts
   - Answers should be concise (typically 1-4 words)

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

class QuestionType(str, Enum):
    FACTUAL = "factual"  # Questions requiring specific fact recall from text

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Simple fact identification (names, dates)
    EASY = 2         # Basic fact recall (events, places)
    MEDIUM = 3       # Multi-part facts (relationships, sequences)
    HARD = 4         # Complex facts (interconnected details)
    VERY_HARD = 5    # Detailed specifics requiring careful reading

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
    text_based: bool = Field(..., description="Answer directly present in text")
    no_tricks: bool = Field(..., description="Avoids misleading phrasing")

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
        default=QuestionType.FACTUAL,
        description="Question type (factual)"
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

1. **Fact Identification**
   - Scan text for specific, verifiable facts
   - Note entities (people, places, organizations)
   - Identify dates, numbers, and quantities
   - Locate technical terms and definitions

2. **Answer Verification**
   - Ensure answer is explicitly stated
   - Confirm answer can be quoted verbatim
   - Verify single correct answer exists
   - Check answer is concise and specific

3. **Question Formation**
   - Create clear "wh-" questions (who, what, when, where)
   - Ensure question targets specific fact
   - Vary question types across comprehension categories
   - Maintain precise language

4. **Quality Verification**
   - Check answer is verbatim from text
   - Verify no external knowledge needed
   - Confirm clear, unambiguous question
   - Test for answer specificity

## Examples

### Example 1: Process Analysis (Easy)

```json
{
    "document_extract_analysis": "The text explains the photosynthesis process in plants, detailing how sunlight is converted into chemical energy.",
    "kind": "factual",
    "testable_concepts": [
        "energy conversion in plants",
        "role of chlorophyll",
        "carbon dioxide utilization",
        "glucose production"
    ],
    "potential_question_directions": [
        "How do plants convert sunlight into usable energy?",
        "What role does chlorophyll play in photosynthesis?",
        "What are the primary inputs and outputs of photosynthesis?",
        "How is glucose formed during this process?"
    ],
    "best_direction": "What is the primary function of chlorophyll in photosynthesis?",
    "comprehension_type": "process_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "What specific role does chlorophyll perform in photosynthesis?",
    "answer": "Chlorophyll captures sunlight and converts it into chemical energy",
    "reasoning": "The text explicitly describes chlorophyll's function in capturing and converting solar energy, which is fundamental to understanding the photosynthesis process",
    "difficulty": 2,
    "difficulty_justification": "While the concept involves a process, the information is clearly stated and requires basic recall of a single function",
    "supporting_quotes": [
        "Chlorophyll molecules in plant cells capture sunlight and convert it into chemical energy",
        "This energy conversion process is the first step in photosynthesis, where light energy becomes usable chemical energy"
    ],
    "quote_context": "These quotes directly establish chlorophyll's role in the energy conversion process, with the first quote stating the specific function and the second quote confirming its place in the broader photosynthesis process"
}
```

### Example 2: System Analysis (Very Hard)

```json
{
    "document_extract_analysis": "The passage details the complex interactions within Earth's climate system, including feedback loops between atmospheric CO2, ocean temperatures, and ice caps.",
    "kind": "factual",
    "testable_concepts": [
        "climate system interactions",
        "feedback mechanisms",
        "carbon cycle",
        "temperature regulation"
    ],
    "potential_question_directions": [
        "How do different components of Earth's climate system interact?",
        "What role do feedback loops play in climate regulation?",
        "How does oceanic CO2 absorption affect global temperature?"
    ],
    "best_direction": "Explain the relationship between oceanic CO2 absorption and global temperature regulation",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How does the ocean's absorption of CO2 influence the global temperature regulation system?",
    "answer": "The oceans absorb excess CO2, which affects their pH levels, altering their capacity to regulate global temperatures through heat absorption and circulation patterns",
    "reasoning": "This answer synthesizes multiple interconnected concepts from the text about oceanic CO2 absorption, pH changes, and temperature regulation mechanisms",
    "difficulty": 5,
    "difficulty_justification": "Requires understanding of multiple interacting systems and their complex relationships, demanding high-level synthesis of information",
    "supporting_quotes": [
        "The oceans act as a carbon sink, absorbing approximately 30% of atmospheric CO2",
        "This absorption leads to ocean acidification, altering pH levels",
        "Changes in ocean chemistry affect global circulation patterns, which are crucial for temperature regulation"
    ],
    "quote_context": "These quotes form a chain of evidence showing how ocean CO2 absorption triggers a cascade of effects impacting global temperature regulation, from initial absorption through chemical changes to circulation impacts"
}
```

### Example 3: Relationship Comprehension (Medium)

```json
{
    "document_extract_analysis": "The text explores the symbiotic relationship between flowering plants and their pollinators, particularly focusing on bees and butterflies.",
    "kind": "factual",
    "testable_concepts": [
        "pollinator behavior",
        "plant adaptations",
        "mutual benefits",
        "evolutionary relationships"
    ],
    "potential_question_directions": [
        "The evolutionary advantages gained through plant-pollinator relationships",
        "The role of specific adaptations in maintaining symbiotic balance",
        "The mechanisms behind successful cross-species cooperation",
        "The impact of pollinator diversity on plant reproduction"
    ],
    "best_direction": "Examining the reciprocal adaptations that maintain the plant-pollinator relationship",
    "comprehension_type": "relationship_comprehension",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "Describe the reciprocal exchange system between flowering plants and their insect pollinators that ensures both species' survival.",
    "answer": "Plants provide nectar and pollen for food, while pollinators facilitate plant reproduction through cross-pollination",
    "reasoning": "The text describes the two-way beneficial relationship between plants and pollinators, showing how each species supports the other's survival through specific adaptations and behaviors",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding and connecting multiple aspects of the relationship between two species, including both the exchange system and its evolutionary significance",
    "supporting_quotes": [
        "Flowering plants produce nectar and protein-rich pollen that serve as essential food sources for bees and butterflies",
        "In return, these insects transfer pollen between flowers, enabling cross-pollination and sexual reproduction in plants"
    ],
    "quote_context": "The quotes explicitly outline both sides of the mutualistic relationship, showing how each species provides benefits to the other through their interactions"
}
```

### Example 4: Evidence Synthesis (Hard)

```json
{
    "document_extract_analysis": "The document presents multiple archaeological findings that support the theory of early human migration patterns across Asia.",
    "kind": "factual",
    "testable_concepts": [
        "archaeological evidence",
        "migration routes",
        "dating methods",
        "cultural artifacts"
    ],
    "potential_question_directions": [
        "What evidence supports the Asian migration theory?",
        "How do different archaeological findings correlate?",
        "What timeline do the artifacts suggest?"
    ],
    "best_direction": "How do multiple archaeological findings support the Asian migration theory?",
    "comprehension_type": "evidence_synthesis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "What combination of archaeological evidence supports the theory of early human migration through Asia?",
    "answer": "The presence of similar stone tools, dated human remains, and genetic evidence from multiple sites along the proposed migration route",
    "reasoning": "The answer combines multiple pieces of evidence presented in the text to support a single theoretical framework",
    "difficulty": 4,
    "difficulty_justification": "Requires synthesis of multiple pieces of evidence and understanding their collective significance",
    "supporting_quotes": [
        "Stone tools found at sites in Mongolia and Siberia share distinctive manufacturing techniques with those discovered in China",
        "Carbon dating of human remains along the proposed route indicates a consistent timeline of migration",
        "DNA analysis of ancient remains shows genetic markers that can be traced from Southeast Asia through Northeast Asia"
    ],
    "quote_context": "These three quotes provide complementary lines of evidence - archaeological artifacts, dating analysis, and genetic data - all supporting the same migration pattern theory"
}
```

### Example 5: Technical Clarification (Medium)

```json
{
    "document_extract_analysis": "The text explains quantum entanglement and its implications for quantum computing.",
    "kind": "factual",
    "testable_concepts": [
        "quantum entanglement principles",
        "particle interdependence",
        "quantum state behavior",
        "distance effects"
    ],
    "potential_question_directions": [
        "Why do quantum particles remain connected?",
        "What role does distance play in entanglement?",
        "How do measurement changes affect entangled particles?",
        "When does quantum entanglement break down?"
    ],
    "best_direction": "Why can't entangled particles be described independently?",
    "comprehension_type": "technical_clarification",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "Why are entangled particles' quantum states impossible to describe independently?",
    "answer": "Because they become inextricably linked, maintaining their connection even when separated by vast distances",
    "reasoning": "The text emphasizes the fundamental characteristic of quantum entanglement where particles maintain their interconnected nature regardless of separation, making independent description impossible",
    "difficulty": 3,
    "difficulty_justification": "While the concept is technical, the explanation of particle interdependence is clearly stated and requires understanding of specific terminology",
    "supporting_quotes": [
        "Quantum entanglement is a phenomenon where two or more particles become inextricably linked",
        "Once entangled, the quantum state of each particle cannot be described independently of the other, even when separated by vast distances"
    ],
    "quote_context": "These quotes establish both the fundamental linking of particles and the persistence of their connection across distances, explaining why independent description is impossible"
}
```

## Common Pitfalls to Avoid

1. **Inferential Questions**
   ❌ "What might have motivated Fleming to study mold?"
   ✅ "What did Fleming discover in the petri dish?"

2. **Vague Questions**
   ❌ "What happened during the Manhattan Project?"
   ✅ "When was the Trinity test conducted?"

3. **Multiple Possible Answers**
   ❌ "What materials were used to build the pyramid?"
   ✅ "How many limestone blocks were used in the Great Pyramid?"

4. **External Knowledge Required**
   ❌ "Why was penicillin important to medicine?"
   ✅ "Who discovered penicillin in 1928?"

## Output Requirements

1. Generate 3-5 factual questions per text extract
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure at least one question is difficulty level 1 or 2
4. Include at least one question at difficulty level 4 or 5
5. All answers must be verbatim quotes from text
6. Questions must target specific, verifiable facts

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