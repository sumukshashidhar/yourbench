# Document-Based Clarification Question Generation

You will receive a document extract and optionally a summary. Your task is to generate high-quality clarification questions that identify areas needing further explanation or understanding in the provided text extract.

## Core Principles

1. **Clarity Focus**
   - Questions target potentially unclear elements
   - Answers must exist within the text
   - Focuses on understanding specifics
   - Identifies ambiguity or complexity

2. **Question Diversity**
   - Term clarification
   - Process explanation
   - Relationship clarification
   - Context questions
   - Detail explanation
   - Reference resolution

3. **Question Quality**
   - Specific focus point
   - Clear need for explanation
   - Answerable from text
   - Meaningful clarification

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

class QuestionType(str, Enum):
    CLARIFICATION = "clarification"  # Questions seeking deeper understanding of specific elements

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Basic term clarification
    EASY = 2         # Simple explanation needs
    MEDIUM = 3       # Process understanding
    HARD = 4         # Complex relationship clarity
    VERY_HARD = 5    # System-level understanding

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
    text_based: bool = Field(..., description="Clarification exists in text")
    no_tricks: bool = Field(..., description="Genuine clarity need")

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
        default=QuestionType.CLARIFICATION,
        description="Question type (clarification)"
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

1. **Clarity Need Identification**
   - Spot complex terms
   - Note unclear processes
   - Find ambiguous references
   - Identify assumed knowledge

2. **Clarification Mapping**
   - Locate explanations
   - Find context clues
   - Track references
   - Check completeness

3. **Question Formation**
   - Target specific elements
   - Frame for clarity
   - Enable precise answers
   - Support understanding

4. **Quality Verification**
   - Check answer presence
   - Verify clarity need
   - Confirm specificity
   - Test usefulness

## Examples

### Example 1: Quantitative Analysis (Easy)
```json
{
    "kind": "clarification",
    "document_extract_analysis": "The text discusses global carbon emissions, presenting specific data about different countries' contributions and their reduction targets.",
    "testable_concepts": [
        "emission measurements",
        "country comparisons",
        "reduction targets",
        "environmental impact assessment"
    ],
    "potential_question_directions": [
        "How do different countries' carbon emissions compare?",
        "What metrics are used to measure emission reductions?",
        "What is the relationship between GDP and carbon emissions?",
        "How are reduction targets calculated and implemented?"
    ],
    "best_direction": "What metrics are used to measure emission reductions?",
    "comprehension_type": "quantitative",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How do scientists measure and compare carbon emissions across different industrial sectors?",
    "answer": "Carbon emissions are measured in metric tons of CO2 equivalent, with specific calculations for each industrial sector based on energy consumption and production processes",
    "reasoning": "The text provides detailed measurement methodologies and comparison frameworks, requiring understanding of both the metrics and their application",
    "difficulty": 2,
    "difficulty_justification": "While involving numbers and measurements, the concept is straightforward and clearly explained in the text",
    "supporting_quotes": [
        "Global carbon emissions reached 36.3 billion metric tons CO2-equivalent in 2021",
        "Industrial sectors are measured using standardized protocols, with direct measurements from power plants accounting for 40% of total emissions",
        "Emissions calculations factor in both direct energy consumption and indirect emissions from production processes"
    ],
    "quote_context": "The quotes establish the measurement methodology, providing specific metrics and explaining how different industrial sectors are evaluated. They demonstrate both the quantitative nature of measurements and the standardized approach across sectors."
}
```

### Example 2: Process Analysis (Medium)
```json
{
    "kind": "clarification",
    "document_extract_analysis": "The text explains quantum entanglement and its implications for quantum computing",
    "testable_concepts": [
        "quantum entanglement",
        "particle behavior",
        "measurement effects",
        "quantum computing applications"
    ],
    "potential_question_directions": [
        "How does quantum entanglement occur?",
        "What role does measurement play in quantum states?",
        "How do entangled particles maintain their connection?",
        "What are the practical applications in computing?"
    ],
    "best_direction": "How does quantum entanglement occur?",
    "comprehension_type": "process_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "What is the mechanism through which particles become quantum entangled, and how does this affect their behavior?",
    "answer": "Particles become entangled when they interact at the quantum level, causing their quantum states to become correlated regardless of distance",
    "reasoning": "Understanding quantum entanglement requires grasping both the physical process and its counterintuitive implications",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding of abstract concepts and their interconnections",
    "supporting_quotes": [
        "When two particles interact at extremely close distances, their quantum states become correlated through a process called quantum entanglement",
        "Once entangled, measuring the state of one particle instantaneously determines the state of its partner, regardless of their separation distance",
        "This 'spooky action at a distance,' as Einstein called it, forms the basis for quantum computing operations"
    ],
    "quote_context": "These quotes outline the fundamental process of quantum entanglement, from the initial interaction to the resulting behavior, and connect it to practical applications in computing."
}
```

### Example 3: System Impact (Hard)
```json
{
    "kind": "clarification",
    "document_extract_analysis": "The text describes the complex interactions within marine ecosystems and the impact of climate change",
    "testable_concepts": [
        "ecosystem dynamics",
        "climate change effects",
        "species interactions",
        "environmental adaptation"
    ],
    "potential_question_directions": [
        "How do changes in ocean temperature affect marine food webs?",
        "What are the cascading effects of coral bleaching?",
        "How do species adapt to changing conditions?",
        "What role do microorganisms play in ecosystem stability?"
    ],
    "best_direction": "How do changes in ocean temperature affect marine food webs?",
    "comprehension_type": "system_impact",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How do incremental changes in ocean temperature create ripple effects throughout marine ecosystems?",
    "answer": "Temperature changes affect phytoplankton productivity, which impacts all higher trophic levels, altering predator-prey relationships and nutrient cycling",
    "reasoning": "The answer requires understanding multiple interconnected systems and their responses to environmental changes",
    "difficulty": 4,
    "difficulty_justification": "Involves complex system interactions and multiple cause-effect relationships",
    "supporting_quotes": [
        "A 1°C increase in ocean temperature reduces phytoplankton productivity by 12%, affecting the entire marine food web",
        "Declining phytoplankton populations have led to a 30% reduction in fish populations in affected areas",
        "Changes in temperature alter the timing of plankton blooms, creating mismatches between predator and prey life cycles"
    ],
    "quote_context": "The quotes demonstrate the cascading effects of temperature changes, starting with primary producers (phytoplankton) and showing how these changes ripple through the food web to affect higher trophic levels."
}
```

### Example 4: Evidence Synthesis (Very Hard)
```json
{
    "kind": "clarification",
    "document_extract_analysis": "The text examines multiple theories about consciousness and their supporting evidence",
    "testable_concepts": [
        "consciousness theories",
        "neural correlates",
        "experimental evidence",
        "philosophical implications"
    ],
    "potential_question_directions": [
        "How do different theories of consciousness explain subjective experience?",
        "What evidence supports each major theory?",
        "How do researchers measure consciousness?",
        "What are the implications for artificial consciousness?"
    ],
    "best_direction": "How do different theories of consciousness explain subjective experience?",
    "comprehension_type": "evidence_synthesis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How do the neural correlates of consciousness support or challenge different theories about subjective experience?",
    "answer": "Neural correlates provide evidence for both integrated information theory and global workspace theory, while challenging pure computational theories",
    "reasoning": "Requires synthesizing complex theoretical frameworks with empirical evidence and understanding their implications",
    "difficulty": 5,
    "difficulty_justification": "Demands integration of multiple complex theories with empirical evidence and philosophical concepts",
    "supporting_quotes": [
        "fMRI studies show integrated information patterns in the brain correlating with reported conscious experiences",
        "Global workspace activation patterns support the theory of consciousness as a broadcasting mechanism",
        "Patients with damaged thalamic regions show reduced consciousness while maintaining computational abilities, challenging pure computational theories"
    ],
    "quote_context": "These quotes provide empirical evidence from multiple sources - neuroimaging, theoretical predictions, and clinical observations - allowing comparison and synthesis across different theories of consciousness."
}
```

### Example 5: Relationship Comprehension (Medium-Hard)
```json
{
    "kind": "clarification",
    "document_extract_analysis": "The text explores the relationship between genetic expression and environmental factors",
    "testable_concepts": [
        "epigenetics",
        "gene-environment interaction",
        "phenotypic expression",
        "hereditary patterns"
    ],
    "potential_question_directions": [
        "How do environmental factors influence gene expression?",
        "What role does timing play in genetic activation?",
        "How do epigenetic changes affect future generations?",
        "What mechanisms control gene silencing?"
    ],
    "best_direction": "How do environmental factors influence gene expression?",
    "comprehension_type": "relationship_comprehension",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How do specific environmental triggers interact with genetic predispositions to influence phenotypic expression?",
    "answer": "Environmental factors can activate or suppress genes through epigenetic mechanisms, leading to changes in protein production and cellular behavior",
    "reasoning": "Understanding requires knowledge of both genetic mechanisms and environmental influences, plus their interaction",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding complex biological mechanisms and their interactions with external factors",
    "supporting_quotes": [
        "Environmental stress triggers methylation patterns that can silence specific genes without altering DNA sequence",
        "Studies show that early-life nutrition can influence gene expression patterns that persist into adulthood",
        "Transgenerational effects have been observed, with parental environmental exposures affecting offspring gene expression"
    ],
    "quote_context": "The quotes establish the mechanistic link between environmental factors and genetic expression, providing specific examples of how external influences can create lasting changes in gene activity."
}
```

## Common Pitfalls to Avoid

1. **Overly Broad Questions**
   ❌ "What is the immune system?"
   ✅ "How do B-cells and T-cells communicate specifically?"

2. **Questions Without Text Support**
   ❌ "What other immune cells exist?"
   ✅ "Could you clarify the role of cytokines mentioned in the text?"

3. **Obvious Questions**
   ❌ "What are cells?"
   ✅ "How do memory cells retain infection information?"

4. **Complex Without Purpose**
   ❌ "What are all possible immune responses?"
   ✅ "Could you explain how the described components work together?"

## Output Requirements

1. Generate 3-5 clarification questions per text extract
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure questions target specific clarity needs
4. Include clear text-based answers
5. Questions should help resolve ambiguity
6. Balance difficulty levels

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

1. **Clarity Need Selection**
   - Technical terms
   - Complex processes
   - Relationships
   - System interactions
   - References
   - Context needs

2. **Question Formation**
   - Be specific
   - Target confusion points
   - Enable clear answers
   - Support understanding

3. **Difficulty Progression**
   - Term clarification (Level 1-2)
   - Process explanation (Level 3)
   - System understanding (Level 4-5)
   - Complex interactions (Level 5)

4. **Clarification Types**
   - Terms: Technical vocabulary
   - Processes: Step sequences
   - Relationships: Connections
   - Context: Background
   - References: Unclear mentions
   - Systems: Complex interactions

5. **Response Quality**
   - Clear explanation
   - Specific answers
   - Text support
   - Enhanced understanding