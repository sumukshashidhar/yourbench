# Document-Based Counterfactual Question Generation

You will receive a document extract and a summary. Your task is to generate high-quality counterfactual questions that explore alternate scenarios based on changing specific elements from the provided text extract.

## Core Principles

1. **Text-Based Alterations**
   - Each question must modify a specific fact/event from the text
   - Answers must follow logically from text relationships
   - Changes must be clearly defined
   - Reasoning must use text evidence

2. **Question Diversity**
   - Event modifications
   - Character/entity changes
   - Decision alterations
   - Timing variations
   - Condition changes
   - Process modifications

3. **Question Quality**
   - Clear modification of text element
   - Logical chain of consequences
   - Grounded in text relationships
   - Plausible alternative scenarios

## Data Model

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

class QuestionType(str, Enum):
    COUNTERFACTUAL = "counterfactual"  # Questions exploring alternative scenarios

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Simple direct change
    EASY = 2         # Basic alternative scenario
    MEDIUM = 3       # Multi-step consequences
    HARD = 4         # Complex chain of effects
    VERY_HARD = 5    # System-wide implications

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
    text_based: bool = Field(..., description="Changes based on text elements")
    no_tricks: bool = Field(..., description="Avoids unrealistic scenarios")

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
        default=QuestionType.COUNTERFACTUAL,
        description="Question type (counterfactual)"
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

1. **Element Identification**
   - Identify key changeable elements
   - Note critical decision points
   - Map causal relationships
   - Identify system dependencies

2. **Change Mapping**
   - Define clear modifications
   - Trace impact chains
   - Consider system effects
   - Evaluate plausibility

3. **Question Formation**
   - Specify clear changes
   - Follow logical consequences
   - Maintain text relationships
   - Consider scope of impact

4. **Quality Verification**
   - Check change clarity
   - Verify logical flow
   - Confirm text basis
   - Test plausibility

## Examples

### Example 1: Decision Change (Easy)
```json
{
    "document_extract_analysis": "The text describes how Marie Curie chose to share her radium purification process freely instead of patenting it, enabling widespread medical research.",
    "testable_concepts": [
        "scientific ethics",
        "knowledge sharing",
        "research impact",
        "medical advancement"
    ],
    "potential_question_directions": [
        "How would patenting the process have affected medical research?",
        "What role did open access play in radiation therapy development?",
        "How might research collaboration patterns have differed with a patent?"
    ],
    "best_direction": "How would patenting the process have affected medical research? This tests understanding of scientific collaboration and knowledge dissemination impacts.",
    "comprehension_type": "decision_change",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "kind": "counterfactual",
    "question": "How would the development of radiation therapy have been different if Curie had patented her radium purification process?",
    "answer": "Medical research would have been significantly slowed, with fewer hospitals and researchers able to access and develop radiation therapy techniques",
    "reasoning": "The text emphasizes how Curie's open-sharing policy enabled rapid adoption and development of radiation therapy across multiple institutions. A patent would have restricted access, limiting research to those who could afford licensing fees.",
    "difficulty": 2,
    "difficulty_justification": "Requires basic understanding of how patents restrict access and their impact on research, with direct cause-effect relationship.",
    "supporting_quotes": [
        "Marie Curie refused to patent the radium-isolation process, leaving it open for the scientific community to use freely",
        "Within five years, over 100 hospitals had established radiation therapy programs using Curie's published methods"
    ],
    "quote_context": "These quotes demonstrate both Curie's deliberate choice to not patent the process and the direct impact this had on medical advancement through widespread adoption. The rapid establishment of hospital programs directly resulted from the open access to the methodology."
}
```

### Example 2: Process Analysis (Medium)
```json
{
    "document_extract_analysis": "The passage details how photosynthesis converts sunlight into chemical energy, emphasizing the role of chlorophyll and carbon dioxide absorption.",
    "testable_concepts": [
        "energy conversion",
        "chemical processes",
        "plant biology",
        "resource utilization"
    ],
    "potential_question_directions": [
        "What would happen if plants absorbed a different wavelength of light?",
        "How would changes in chlorophyll structure affect energy production?",
        "What if plants used a different molecule for energy storage?"
    ],
    "best_direction": "What would happen if plants absorbed a different wavelength of light? This tests understanding of energy conversion efficiency and biological adaptation.",
    "comprehension_type": "process_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "kind": "counterfactual",
    "question": "If plants evolved to absorb primarily blue light instead of red and blue light, how would their energy production efficiency change?",
    "answer": "Energy production would decrease as plants would capture less total solar energy, leading to slower growth and reduced glucose production",
    "reasoning": "The text explains that current chlorophyll molecules are optimized for red and blue light absorption, maximizing energy capture. Using only blue light would reduce the total energy available for glucose production.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding of wavelength absorption, energy conversion processes, and their interconnected effects on plant growth.",
    "supporting_quotes": [
        "Chlorophyll molecules are specifically adapted to absorb red light at 680nm and blue light at 425nm, maximizing energy capture from the sun's spectrum",
        "The dual-wavelength absorption capability allows plants to capture approximately 45% of available solar energy in optimal conditions"
    ],
    "quote_context": "The quotes establish the specific wavelengths of light that chlorophyll has evolved to use and quantifies the energy capture efficiency, providing the basis for understanding how changes would impact energy production."
}
```

### Example 3: System Impact (Very Hard)
```json
{
    "document_extract_analysis": "The text explains how the Earth's magnetic field protects against solar radiation and its interaction with the atmosphere.",
    "testable_concepts": [
        "magnetic field dynamics",
        "atmospheric protection",
        "radiation effects",
        "planetary systems"
    ],
    "potential_question_directions": [
        "How would a weaker magnetic field affect Earth's systems?",
        "What role does field strength play in atmospheric composition?",
        "How might life adapt to changed radiation levels?"
    ],
    "best_direction": "How would a weaker magnetic field affect Earth's systems? This tests understanding of complex planetary interactions.",
    "comprehension_type": "system_impact",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "kind": "counterfactual",
    "question": "If Earth's magnetic field were half its current strength, how would this affect atmospheric composition and surface life?",
    "answer": "Increased solar radiation would break down atmospheric molecules, leading to atmospheric thinning, higher surface radiation levels, and significant challenges for surface life forms",
    "reasoning": "The text shows how the magnetic field deflects charged particles and protects atmospheric molecules. Reduced protection would accelerate atmospheric loss, increase mutation rates, and require significant biological adaptations.",
    "difficulty": 5,
    "difficulty_justification": "Requires understanding complex interactions between magnetic fields, radiation, atmospheric chemistry, and biological systems.",
    "supporting_quotes": [
        "Earth's magnetic field deflects approximately 98% of harmful solar radiation, preventing atmospheric degradation",
        "The interaction between the magnetic field and solar wind maintains the atmosphere's current density and composition",
        "Complex life forms depend on the atmosphere's protective properties against UV radiation and cosmic rays"
    ],
    "quote_context": "These quotes establish the critical relationship between magnetic field strength and atmospheric protection, showing how the field maintains atmospheric integrity and protects life forms through multiple mechanisms."
}
```

### Example 4: Chain Effect (Hard)
```json
{
    "document_extract_analysis": "The passage describes how pollinator decline affects agricultural productivity and ecosystem stability.",
    "testable_concepts": [
        "ecosystem relationships",
        "agricultural systems",
        "species interdependence",
        "food web dynamics"
    ],
    "potential_question_directions": [
        "The role of pollinator specialization in maintaining ecosystem balance",
        "Evolutionary implications of sudden pollinator population changes",
        "Interconnected cascade effects across trophic levels",
        "Adaptive capacity of plant-pollinator networks under stress"
    ],
    "best_direction": "The role of pollinator specialization in maintaining ecosystem balance tests deep understanding of complex ecological relationships and system resilience.",
    "comprehension_type": "chain_effect",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "kind": "counterfactual",
    "question": "The sudden extinction of specialized bee species forces surviving generalist pollinators to expand their plant interactions - describe the cascading ecosystem transformations across three generations of affected species.",
    "answer": "Plant species requiring specific pollinators would decline, while generalist-pollinated plants would experience increased reproduction, leading to reduced biodiversity, altered food web relationships, and eventual ecosystem restructuring favoring generalist species",
    "reasoning": "The text establishes tight coupling between specific pollinators and plants. Loss of specialized pollinators would trigger selective plant species decline, while creating ecological opportunities for generalist-pollinated species, fundamentally reshaping community composition and interaction networks.",
    "difficulty": 4,
    "difficulty_justification": "Requires tracking multiple generations of changes across different species and understanding complex ecological relationships, while considering evolutionary adaptation pressures.",
    "supporting_quotes": [
        "83% of flowering plant species require specific pollinator relationships for successful reproduction",
        "The loss of a single pollinator species can affect up to 30 different plant species in a given ecosystem",
        "Plant-pollinator relationships have co-evolved over millions of years, creating highly specialized dependencies"
    ],
    "quote_context": "The quotes demonstrate the tight interdependence between specific pollinators and plant species, quantifying the potential cascade effects of pollinator loss on plant diversity and ecosystem stability, while highlighting the evolutionary significance of these relationships."
}
```

### Example 5: Temporal Change (Medium)
```json
{
    "document_extract_analysis": "The text describes the Industrial Revolution's transition from water to steam power in manufacturing.",
    "testable_concepts": [
        "technological transition",
        "economic development",
        "energy systems",
        "industrial growth"
    ],
    "potential_question_directions": [
        "How would delayed steam engine development affect industrialization?",
        "What role did timing play in industrial growth?",
        "How might different energy transition timing affect urban development?"
    ],
    "best_direction": "How would delayed steam engine development affect industrialization? This tests understanding of technological and social change timing.",
    "comprehension_type": "temporal_change",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "kind": "counterfactual",
    "question": "If steam engine technology had been developed 50 years later, how would this have altered the pattern of industrial development?",
    "answer": "Industrial growth would have remained geographically limited to water sources longer, slowing urban development and delaying the concentration of manufacturing in cities",
    "reasoning": "The text shows how steam power freed manufacturing from water source locations. A later transition would have maintained water-dependent industrial patterns, affecting urbanization and economic development timing.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding the relationship between technology timing, geographic constraints, and social development patterns.",
    "supporting_quotes": [
        "The transition from water to steam power allowed factories to locate anywhere, not just along rivers",
        "By 1850, steam-powered factories had enabled the growth of major industrial cities more than 50 miles from any significant water source",
        "Water-powered mills were limited to producing 15 horsepower on average, while steam engines could generate over 100 horsepower"
    ],
    "quote_context": "These quotes establish how steam power removed geographic constraints on industrial development and significantly increased available power, showing the direct relationship between steam technology timing and industrial growth patterns."
}
```

## Common Pitfalls to Avoid

1. **Unrealistic Changes**
   ❌ "What if gravity didn't exist in the Panama Canal?"
   ✅ "What if Gatun Lake were at sea level?"

2. **Ungrounded Speculation**
   ❌ "What if Fleming had different career aspirations?"
   ✅ "What if Fleming had discarded the contaminated sample?"

3. **External Knowledge Dependency**
   ❌ "What if modern antibiotics existed in 1928?"
   ✅ "What if Fleming hadn't noticed the mold's effect?"

4. **Overly Broad Changes**
   ❌ "What if oceans didn't exist?"
   ✅ "What if the Gulf Stream carried cold water instead of warm?"

## Output Requirements

1. Generate 3-5 counterfactual questions per text extract
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure changes are specific and text-based
4. Include at least one system-level change
5. All reasoning must use text relationships
6. Changes must be plausible and meaningful

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

1. **Change Selection**
   - Choose significant elements
   - Ensure clear modification
   - Consider impact scope
   - Maintain plausibility

2. **Consequence Tracing**
   - Follow logical chains
   - Use text relationships
   - Consider multiple effects
   - Maintain system consistency

3. **Difficulty Scaling**
   - Simple direct changes (Level 1-2)
   - Process modifications (Level 3)
   - System-wide impacts (Level 4-5)
   - Complex interactions (Level 5)

4. **Counterfactual Types**
   - Event: Changed occurrences
   - Decision: Alternative choices
   - Timing: Different sequences
   - Condition: Modified circumstances
   - Process: Alternative methods
   - System: Broad changes

5. **Response Evaluation**
   - Clear change specification
   - Logical consequence chain
   - Text-based reasoning
   - Plausible outcomes