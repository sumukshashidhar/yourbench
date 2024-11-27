# Document-Based Analytical Question Generation

You will receive a document extract and a summary. Your task is to generate high-quality analytical questions that test deeper understanding, relationships, and implications from the provided text extract.

## Core Principles

1. **Evidence-Based Analysis**
   - Questions must be answerable using the text's content
   - Answers should synthesize information from multiple parts of the text
   - No external knowledge requirements
   - Requires understanding relationships and implications

2. **Question Diversity**
   - Compare and contrast elements
   - Cause and effect relationships
   - Pattern identification
   - Process analysis
   - Evidence evaluation
   - Argument analysis

3. **Question Quality**
   - Clear analytical focus
   - Multiple text elements involved
   - Requires synthesis of information
   - Tests deeper understanding

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

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
        default=QuestionType.ANALYTICAL,
        description="Question type (analytical)"
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

1. **Relationship Identification**
   - Identify cause-effect relationships
   - Note comparison opportunities
   - Map processes and systems
   - Identify patterns and trends

2. **Evidence Mapping**
   - Locate relevant text passages
   - Connect related information
   - Identify supporting details
   - Map information flow

3. **Question Formation**
   - Focus on relationships and patterns
   - Require information synthesis
   - Target analytical thinking
   - Structure for clarity

4. **Quality Verification**
   - Verify evidence sufficiency
   - Check analytical depth
   - Confirm text-based answering
   - Test for clarity

## Examples

### Example 1: Pattern Recognition (Easy)
```json
{
    "document_extract_analysis": "The text discusses migration patterns of monarch butterflies, including their navigation methods and seasonal timing.",
    "kind": "analytical",
    "testable_concepts": [
        "navigation mechanisms",
        "seasonal triggers",
        "generational memory",
        "environmental cues"
    ],
    "potential_question_directions": [
        "Why do monarchs rely on multiple navigation systems?",
        "What role does generational memory play in migration?",
        "Why do specific seasonal changes trigger migration?",
        "Why is coordination across generations essential?"
    ],
    "best_direction": "Why is the combination of environmental cues and generational memory critical for monarch migration?",
    "comprehension_type": "pattern_recognition",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "Why do monarch butterflies require both environmental cues and genetic programming to maintain their migration patterns?",
    "answer": "Monarchs use a combination of environmental cues (sunlight position, magnetic fields) and inherited genetic programming to maintain consistent migration routes",
    "reasoning": "The text shows how multiple factors work together - environmental sensing, genetic programming, and timing mechanisms all contribute to successful migration patterns across generations",
    "difficulty": 2,
    "difficulty_justification": "Requires connecting multiple concepts but follows a clear pattern described in the text",
    "supporting_quotes": [
        "Monarchs use the position of the sun as a primary navigational cue, adjusting their internal compass throughout the day",
        "Genetic studies have revealed that inherited DNA sequences encode timing mechanisms and directional preferences",
        "Environmental factors including day length and temperature trigger hormonal changes that initiate migration"
    ],
    "quote_context": "These quotes demonstrate the dual nature of monarch navigation - both immediate environmental sensing and inherited genetic programming. The first quote shows active navigation, while the second and third quotes reveal the underlying biological mechanisms that enable this behavior across generations."
}
```

### Example 2: Process Analysis (Medium)
```json
{
    "document_extract_analysis": "The passage explains how neurons communicate through synaptic transmission using neurotransmitters and electrical signals.",
    "kind": "analytical",
    "testable_concepts": [
        "synaptic transmission",
        "neurotransmitter release",
        "receptor activation",
        "signal propagation"
    ],
    "potential_question_directions": [
        "How does calcium influence neurotransmitter release?",
        "What role do receptor proteins play in signal transmission?",
        "How is the electrical signal converted to chemical transmission?",
        "What mechanisms ensure precise timing of neural communication?"
    ],
    "best_direction": "How does the conversion between electrical and chemical signals enable neural communication?",
    "comprehension_type": "process_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How do neurons maintain precise timing in signal transmission while converting between electrical and chemical signals?",
    "answer": "Neurons use voltage-gated calcium channels to trigger neurotransmitter release, coordinating electrical and chemical signals through precise molecular mechanisms",
    "reasoning": "The text describes the complex interplay between electrical signals, calcium influx, vesicle release, and receptor activation, showing how timing is maintained through molecular mechanisms",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding multiple steps and their precise coordination in a complex biological process",
    "supporting_quotes": [
        "Action potentials trigger the opening of voltage-gated calcium channels in the presynaptic terminal",
        "The influx of calcium ions causes synaptic vesicles to fuse with the membrane within microseconds",
        "Neurotransmitter molecules bind to specific receptor proteins, initiating a new electrical signal in the postsynaptic neuron"
    ],
    "quote_context": "These quotes trace the precise sequence of neural signal transmission, from electrical signal to chemical release and back to electrical. They highlight the molecular mechanisms that maintain precise timing throughout the process."
}
```

### Example 3: System Impact (Hard)
```json
{
    "document_extract_analysis": "The text examines how deforestation affects global carbon cycles and local ecosystems simultaneously.",
    "kind": "analytical",
    "testable_concepts": [
        "carbon cycle disruption",
        "ecosystem cascade effects",
        "biodiversity impact",
        "climate feedback loops"
    ],
    "potential_question_directions": [
        "How does deforestation create multiple feedback loops?",
        "What are the cascading effects on local ecosystems?",
        "How do carbon cycle changes affect global systems?",
        "What connections exist between local and global impacts?"
    ],
    "best_direction": "How do local ecosystem changes from deforestation connect to global climate effects?",
    "comprehension_type": "system_impact",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "How do the immediate effects of deforestation create long-term changes in both local ecosystems and global climate systems?",
    "answer": "Deforestation triggers immediate biodiversity loss and soil degradation locally, while reducing carbon storage capacity and altering weather patterns globally",
    "reasoning": "The text demonstrates how local ecosystem destruction creates ripple effects through multiple systems, connecting immediate habitat loss to long-term climate changes through various mechanisms",
    "difficulty": 4,
    "difficulty_justification": "Requires analyzing complex interactions between multiple systems across different scales and timeframes",
    "supporting_quotes": [
        "Local deforestation immediately reduces habitat complexity, leading to the loss of 75-80% of resident species within months",
        "Soil erosion accelerates after tree removal, reducing agricultural productivity by 60% within 3 years",
        "Each hectare of forest loss reduces carbon storage capacity by 250 metric tons while releasing stored carbon",
        "Changes in local evapotranspiration patterns affect regional rainfall, creating feedback loops that extend beyond the deforested area"
    ],
    "quote_context": "These quotes connect immediate local impacts to broader systemic effects, showing how deforestation creates cascading changes across multiple scales. They demonstrate both the direct consequences and the indirect feedback mechanisms that amplify the impact."
}
```

### Example 4: Evidence Synthesis (Very Hard)
```json
{
    "document_extract_analysis": "The passage details how quantum entanglement enables quantum computing advantages through parallel processing.",
    "kind": "analytical",
    "testable_concepts": [
        "quantum entanglement",
        "superposition states",
        "parallel computation",
        "quantum algorithms"
    ],
    "potential_question_directions": [
        "Where in quantum systems do computational bottlenecks occur?",
        "In which scenarios do quantum advantages break down?",
        "Where do classical and quantum computations diverge in performance?",
        "At what points do error correction mechanisms become critical?"
    ],
    "best_direction": "Where in quantum computing systems do entanglement and superposition create the most significant advantages over classical computers?",
    "comprehension_type": "evidence_synthesis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "Where in computational processes do quantum entanglement and superposition provide the most dramatic advantages over classical computing methods?",
    "answer": "The most significant advantages appear in parallel processing scenarios where entangled qubits can represent exponentially more states (2^n) than classical bits, particularly in algorithms requiring simultaneous state calculations",
    "reasoning": "The text demonstrates that quantum advantages are most pronounced in specific computational contexts where the exponential scaling of quantum states through entanglement and superposition can be effectively leveraged, particularly in parallel processing scenarios",
    "difficulty": 5,
    "difficulty_justification": "Requires synthesizing complex quantum concepts and identifying specific computational contexts where quantum advantages manifest",
    "supporting_quotes": [
        "Quantum entanglement allows multiple qubits to share quantum states, enabling instantaneous correlation across the system",
        "Through superposition, each qubit exists in multiple states simultaneously, allowing n qubits to represent 2^n states",
        "The combination of entanglement and superposition enables quantum algorithms to perform certain calculations exponentially faster than classical computers",
        "Quantum error correction maintains coherence by distributing quantum information across multiple entangled qubits"
    ],
    "quote_context": "These quotes pinpoint the specific computational scenarios where quantum advantages emerge. They illustrate how the combination of entanglement and superposition creates exponential advantages in state representation and parallel processing, while also indicating where error correction becomes necessary."
}
```

### Example 5: Mechanism Understanding (Medium-Hard)
```json
{
    "document_extract_analysis": "The text explains how CRISPR gene editing achieves precise DNA modifications through molecular mechanisms.",
    "kind": "analytical",
    "testable_concepts": [
        "guide RNA targeting",
        "DNA cutting mechanism",
        "repair pathways",
        "precision control"
    ],
    "potential_question_directions": [
        "What makes guide RNA targeting so precise?",
        "What controls the DNA cutting process?",
        "What role do repair mechanisms play?",
        "What factors influence editing efficiency?"
    ],
    "best_direction": "What molecular mechanisms work together to ensure precise gene editing?",
    "comprehension_type": "mechanism_understanding",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "question": "What molecular components of CRISPR coordinate to achieve precise genetic modifications while minimizing errors?",
    "answer": "CRISPR combines specific guide RNA targeting, controlled Cas9 cutting, and cellular repair mechanisms to make precise DNA changes while preventing off-target effects",
    "reasoning": "The text describes how multiple molecular mechanisms (RNA guidance, protein cutting, and DNA repair) coordinate to achieve accurate genetic modifications",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding multiple molecular mechanisms and their precise coordination in a complex biological process",
    "supporting_quotes": [
        "The guide RNA sequence contains 20 nucleotides that match the target DNA sequence with high specificity",
        "Cas9 protein creates a precise double-strand break only when the guide RNA forms a complete match with the target",
        "Cellular DNA repair mechanisms either join broken ends directly or use a template to insert new sequences",
        "Off-target effects are minimized through careful guide RNA design and controlled Cas9 activity levels"
    ],
    "quote_context": "These quotes outline the sequential steps of CRISPR gene editing, showing how molecular specificity is maintained at each stage. They demonstrate the multiple layers of control that ensure precise genetic modifications while preventing unwanted changes."
}
```

## Common Pitfalls to Avoid

1. **Surface-Level Questions**
   ❌ "What is photosynthesis?"
   ✅ "How do the stages of photosynthesis work together to produce glucose?"

2. **Single-Factor Focus**
   ❌ "Does ice reflect sunlight?"
   ✅ "How does ice reflection contribute to climate feedback loops?"

3. **Requiring External Knowledge**
   ❌ "Why is the greenhouse effect important for Earth?"
   ✅ "How do the described feedback mechanisms affect Arctic temperatures?"

4. **Oversimplification**
   ❌ "What happens when ice melts?"
   ✅ "How does ice melt create a self-reinforcing cycle in the climate system?"

## Output Requirements

1. Generate 3-5 analytical questions per text extract
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure questions require synthesis of multiple text elements
4. Include at least one complex system analysis
5. All reasoning must be supported by text evidence
6. Questions should probe relationships and patterns

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

1. **Question Formation**
   - Begin with "how" and "why" rather than "what"
   - Focus on relationships between elements
   - Require synthesis of multiple pieces of evidence
   - Target understanding of mechanisms and systems

2. **Evidence Use**
   - Identify multiple relevant quotes
   - Show connections between quotes
   - Support analysis with text evidence
   - Demonstrate relationship patterns

3. **Difficulty Progression**
   - Simple relationships (Level 1-2)
   - Multi-factor analysis (Level 3)
   - Complex systems (Level 4-5)
   - Interconnected patterns (Level 5)

4. **Analysis Types**
   - Compare/Contrast: Similarities and differences
   - Cause/Effect: Chains of causation
   - Process: Steps and mechanisms
   - Systems: Interconnected elements
   - Patterns: Recurring themes or relationships
   - Arguments: Evidence and conclusions

5. **Response Evaluation**
   - Logical connection of elements
   - Support from multiple quotes
   - Clear analytical reasoning
   - Demonstrated understanding of relationships