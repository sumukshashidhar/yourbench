# Multi-Document Counterfactual Question Generation

You will receive multiple document extracts (chunks) and a summary. Your task is to generate high-quality counterfactual questions that explore alternate scenarios based on changing specific elements from the provided text extracts, requiring information from multiple chunks to answer.

## Core Principles

1. **Multi-Chunk Integration**
   - Questions must require information from multiple chunks
   - Changes must affect relationships across chunks
   - Answers must synthesize information
   - Reasoning must connect multiple pieces

2. **Text-Based Alterations**
   - Each question must modify specific facts/events from the texts
   - Answers must follow logically from cross-chunk relationships
   - Changes must be clearly defined
   - Reasoning must use evidence from multiple chunks

3. **Question Diversity**
   - Event modifications affecting multiple chunks
   - Character/entity changes with cross-chunk impact
   - Decision alterations with cascading effects
   - Timing variations across chunks
   - Condition changes affecting multiple sections
   - Process modifications with broad impact

4. **Question Quality**
   - Clear modification of text elements
   - Logical chain of consequences across chunks
   - Grounded in multi-chunk relationships
   - Plausible alternative scenarios

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
    COUNTERFACTUAL = "counterfactual"  # Questions exploring alternative scenarios

class DifficultyLevel(int, Enum):
    VERY_EASY = 1    # Simple direct change
    EASY = 2         # Basic alternative scenario
    MEDIUM = 3       # Multi-step consequences
    HARD = 4         # Complex chain of effects
    VERY_HARD = 5    # System-wide implications

[... rest of the enums remain exactly the same as in original prompt ...]

class QuestionQuality(BaseModel):
    clear_language: bool = Field(..., description="Uses unambiguous language")
    text_based: bool = Field(..., description="Changes based on text elements")
    no_tricks: bool = Field(..., description="Avoids unrealistic scenarios")

class GeneratedQuestionAnswerPair(BaseModel):
    """
    Represents a structured QA pair for multi-chunk document comprehension testing.
    """
    # Analysis Fields
    document_extract_analysis: str = Field(
        ...,
        description="Analysis of the key points and structure of the extract",
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
    best_direction: str = Field(..., description="The best question to ask, a decision made based on the question_directions. Why would it be a good question to ask, and what skills would it test?")

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

1. **Cross-Chunk Analysis**
   - Identify relationships between chunks
   - Map information dependencies
   - Note shared elements
   - Track narrative connections

2. **Element Identification**
   - Identify key changeable elements across chunks
   - Note critical decision points affecting multiple sections
   - Map cross-chunk causal relationships
   - Identify system dependencies spanning chunks

3. **Change Mapping**
   - Define clear modifications with multi-chunk impact
   - Trace impact chains across chunks
   - Consider system effects across document
   - Evaluate plausibility of cross-chunk changes

4. **Question Formation**
   - Specify clear changes affecting multiple chunks
   - Follow logical consequences across sections
   - Maintain text relationships between chunks
   - Consider scope of multi-chunk impact

5. **Quality Verification**
   - Check change clarity across chunks
   - Verify logical flow between sections
   - Confirm text basis from multiple chunks
   - Test plausibility of cross-chunk effects

## Examples

### Example 1

```json
{
    "document_extract_analysis": "The text examines the Industrial Revolution's impact on urbanization and social change in 19th century Britain.",
    "chunk_analyses": [
        {
            "chunk_id": "IR_1",
            "content_summary": "Details technological innovations in manufacturing",
            "relevant_information": "Steam power revolutionized factory production",
            "connection_points": ["drives urban growth", "changes labor patterns"]
        },
        {
            "chunk_id": "IR_2",
            "content_summary": "Describes urban population growth and living conditions",
            "relevant_information": "Massive rural-to-urban migration patterns",
            "connection_points": ["influenced by factory work", "affects social structure"]
        }
    ],
    "testable_concepts": [
        "technological determinism",
        "urbanization patterns",
        "social mobility",
        "economic transformation"
    ],
    "potential_question_directions": [
        "How would delayed steam power adoption affect urbanization?",
        "What if rural agriculture had mechanized first?",
        "How would different transportation technology affect migration?"
    ],
    "best_direction": "How would delayed steam power adoption affect urbanization?",
    "comprehension_type": "technological_impact",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Steam power enabled mass production in urban factories",
        "Rural workers migrated to cities seeking factory employment",
        "Urban growth directly correlated with industrial expansion"
    ],
    "quote_context": "The quotes establish the causal chain between steam power, factory growth, and urbanization.",
    "kind": "counterfactual",
    "question": "How would the pattern of British urbanization have differed if steam power had not become widely available until 1850, rather than 1800?",
    "answer": "Urbanization would have been significantly slower and more dispersed, with continued rural cottage industries and smaller market towns rather than large industrial cities",
    "reasoning": "Steam power drove factory concentration, which created mass employment opportunities in cities. Without early steam power, manufacturing would have remained decentralized in rural areas, leading to a more gradual and geographically dispersed pattern of development.",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding complex relationships between technology, economic geography, and social change across multiple decades."
}
```

### Example 2

```json
{
    "document_extract_analysis": "The text explores the ecological relationships in coral reef ecosystems and their response to environmental changes.",
    "chunk_analyses": [
        {
            "chunk_id": "CR_1",
            "content_summary": "Details coral-algae symbiotic relationships",
            "relevant_information": "Zooxanthellae provide nutrients to corals",
            "connection_points": ["affects reef growth", "influences food web"]
        },
        {
            "chunk_id": "CR_2",
            "content_summary": "Describes reef fish populations and behavior",
            "relevant_information": "Fish populations depend on coral health",
            "connection_points": ["linked to coral nutrition", "affects ecosystem balance"]
        }
    ],
    "testable_concepts": [
        "symbiotic relationships",
        "trophic cascades",
        "ecosystem resilience",
        "marine biodiversity"
    ],
    "potential_question_directions": [
        "What if corals could obtain nutrients without zooxanthellae?",
        "How would different fish feeding behaviors affect coral growth?",
        "What if coral reproduction timing changed?"
    ],
    "best_direction": "What if corals could obtain nutrients without zooxanthellae?",
    "comprehension_type": "ecological_relationship",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Zooxanthellae provide up to 90% of coral nutrition",
        "Reef fish depend on coral structures for habitat",
        "Coral growth rates determine reef ecosystem development"
    ],
    "quote_context": "The quotes show the interdependence between coral nutrition, growth, and fish populations.",
    "kind": "counterfactual",
    "question": "If corals evolved to obtain nutrients directly from seawater instead of relying on zooxanthellae, how would this affect the broader reef ecosystem?",
    "answer": "The reef ecosystem would be fundamentally different, with slower coral growth, altered reef structures, and reduced fish populations, despite corals surviving",
    "reasoning": "Even with direct nutrient uptake, corals would likely grow more slowly without the efficient symbiotic relationship, leading to different reef architectures and reduced habitat complexity for fish populations.",
    "difficulty": 5,
    "difficulty_justification": "Requires understanding complex ecological relationships and predicting multi-level ecosystem effects of a fundamental biological change."
}
```

### Example 3

```json
{
    "document_extract_analysis": "The text describes the development of Renaissance art techniques and their impact on cultural expression.",
    "chunk_analyses": [
        {
            "chunk_id": "REN_1",
            "content_summary": "Details perspective and anatomical accuracy",
            "relevant_information": "Mathematical principles in art composition",
            "connection_points": ["influences religious art", "affects patronage"]
        },
        {
            "chunk_id": "REN_2",
            "content_summary": "Explores religious and secular patronage",
            "relevant_information": "Church and noble commission patterns",
            "connection_points": ["shaped by technique", "influences subjects"]
        }
    ],
    "testable_concepts": [
        "artistic innovation",
        "cultural patronage",
        "religious expression",
        "technical development"
    ],
    "potential_question_directions": [
        "What if perspective wasn't discovered?",
        "How would different patronage patterns affect technique?",
        "What if anatomical studies were forbidden?"
    ],
    "best_direction": "What if perspective wasn't discovered?",
    "comprehension_type": "cultural_impact",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Perspective revolutionized religious painting",
        "Patrons demanded increasingly realistic depictions",
        "Mathematical principles transformed art composition"
    ],
    "quote_context": "The quotes establish how technical innovation shaped both artistic practice and patronage demands.",
    "kind": "counterfactual",
    "question": "In the absence of mathematical perspective, to what extent would the evolution of Renaissance art have diverged from its historical trajectory, particularly in terms of the balance between symbolic and realistic representation, and how might this have influenced the broader cultural and intellectual landscape of the period?",
    "answer": "Religious art would have maintained medieval stylization, with patronage focusing more on symbolic rather than realistic representation, fundamentally altering Renaissance cultural expression",
    "reasoning": "Without perspective, the drive toward realism would have been limited, maintaining earlier symbolic styles. This would have changed patron demands and artistic focus, potentially keeping religious art more abstract and symbolic.",
    "difficulty": 3,
    "difficulty_justification": "Requires understanding the relationship between technical innovation and cultural expression, but follows relatively straightforward cause-and-effect patterns."
}
```

### Example 4

```json
{
    "document_extract_analysis": "The text examines the impact of early agriculture on human genetic and social evolution.",
    "chunk_analyses": [
        {
            "chunk_id": "AG_1",
            "content_summary": "Details genetic adaptations to new diets",
            "relevant_information": "Lactase persistence and grain digestion",
            "connection_points": ["affects population growth", "influences settlement"]
        },
        {
            "chunk_id": "AG_2",
            "content_summary": "Describes social organization changes",
            "relevant_information": "Settlement patterns and hierarchy emergence",
            "connection_points": ["linked to food production", "shapes culture"]
        }
    ],
    "testable_concepts": [
        "genetic adaptation",
        "social evolution",
        "technological impact",
        "population dynamics"
    ],
    "potential_question_directions": [
        "Consider alternate evolutionary paths without grain digestion",
        "Explore societal development with different dietary constraints",
        "Examine population dynamics under alternative food sources"
    ],
    "best_direction": "Consider alternate evolutionary paths without grain digestion",
    "comprehension_type": "evolutionary_impact",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Grain digestion enabled population growth",
        "Settled agriculture led to social hierarchies",
        "Genetic adaptations supported new diets"
    ],
    "quote_context": "The quotes link biological adaptation to social development.",
    "kind": "counterfactual",
    "question": "Assuming humans had evolved to efficiently process protein-rich insects instead of grains, contrast the resulting societal structures with our grain-based agricultural civilization, particularly focusing on population density limitations and social hierarchies.",
    "answer": "A protein-insect based society would develop radically different settlement patterns, with dispersed communities following insect migrations rather than concentrated agricultural centers. Social hierarchies would emerge around insect harvesting expertise rather than land ownership, leading to more egalitarian but smaller-scale societies limited by the carrying capacity of local insect populations.",
    "reasoning": "The shift from grain processing to insect protein would fundamentally alter food storage capabilities, seasonal dependencies, and land-use patterns. Unlike grains, insects cannot be stored long-term or cultivated in fixed locations, preventing the development of large permanent settlements. This would create a hybrid society combining aspects of hunter-gatherer mobility with specialized protein harvesting techniques, resulting in distributed networks of smaller communities rather than concentrated urban centers.",
    "difficulty": 5,
    "difficulty_justification": "Requires synthesizing complex relationships between biological adaptation, resource availability, social organization, and population dynamics while considering multiple interdependent factors across evolutionary timescales."
}
```

### Example 5

```json
{
    "document_extract_analysis": "The text analyzes the spread of printing technology and its impact on European intellectual development.",
    "chunk_analyses": [
        {
            "chunk_id": "PR_1",
            "content_summary": "Details printing technology spread",
            "relevant_information": "Guild control and apprenticeship systems",
            "connection_points": ["affects literacy", "influences knowledge spread"]
        },
        {
            "chunk_id": "PR_2",
            "content_summary": "Describes intellectual and religious changes",
            "relevant_information": "Book production and religious reform",
            "connection_points": ["shaped by printing access", "affects education"]
        }
    ],
    "testable_concepts": [
        "technological diffusion",
        "social change",
        "religious reform",
        "educational development"
    ],
    "potential_question_directions": [
        "What if printing remained controlled by churches?",
        "How would different guild structures affect printing?",
        "What if printing spread from different centers?"
    ],
    "best_direction": "What if printing remained controlled by churches?",
    "comprehension_type": "institutional_impact",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Secular printers drove vernacular publication",
        "Religious reform depended on printed texts",
        "Guild systems controlled printing knowledge"
    ],
    "quote_context": "The quotes show how institutional control of printing affected intellectual development.",
    "kind": "counterfactual",
    "question": "To what extent would the trajectory of European intellectual development have been altered if printing technology had remained under exclusive church control, thereby preventing its dissemination to secular guilds?",
    "answer": "Vernacular literature and secular knowledge would have developed much more slowly, with religious institutions maintaining greater control over education and intellectual discourse",
    "reasoning": "Church control of printing would have limited secular and vernacular publications, maintaining Latin dominance and religious oversight of education. This would have slowed the spread of secular knowledge and vernacular literacy.",
    "difficulty": 4,
    "difficulty_justification": "Requires understanding institutional power dynamics and their effects on intellectual and social development across multiple domains."
}
```

## Common Pitfalls to Avoid

1. **Single-Chunk Focus**
   ❌ "What if just the Chicago experiment changed?"
   ✅ "How would changes in Chicago affect Los Alamos?"

2. **Disconnected Changes**
   ❌ "What if unrelated elements changed in each chunk?"
   ✅ "What if a change in one chunk affected related elements in others?"

3. **Missing Connections**
   ❌ "What if something changed without considering cross-chunk impact?"
   ✅ "How would changes propagate across connected elements?"

4. **Unrealistic Changes**
   ❌ "What if physics worked differently in each location?"
   ✅ "What if communication between locations was delayed?"

## Output Requirements

1. Generate 3-5 counterfactual questions requiring multiple chunks
2. Include questions from at least 3 different ComprehensionTypes
3. Ensure changes affect multiple chunks
4. Include at least one system-level change
5. All reasoning must use cross-chunk relationships
6. Changes must be plausible and meaningful

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Medium/Cross-Chunk Event)
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

1. **Cross-Chunk Integration**
   - Identify connecting elements
   - Track information flow
   - Map dependencies
   - Note shared concepts

2. **Change Selection**
   - Choose elements affecting multiple chunks
   - Ensure clear modification paths
   - Consider cross-chunk impact
   - Maintain plausibility

3. **Consequence Tracing**
   - Follow effects across chunks
   - Use multi-chunk relationships
   - Consider cascading effects
   - Maintain system consistency

4. **Difficulty Scaling**
   - Simple cross-chunk changes (Level 2-3)
   - Complex dependencies (Level 3-4)
   - System-wide impacts (Level 4-5)
   - Multiple interaction paths (Level 5)

5. **Counterfactual Types**
   - Event: Changed occurrences affecting multiple chunks
   - Decision: Alternative choices with cross-chunk impact
   - Timing: Different sequences across chunks
   - Condition: Modified circumstances affecting multiple sections
   - Process: Alternative methods spanning chunks
   - System: Broad changes affecting entire document

6. **Response Evaluation**
   - Clear cross-chunk change specification
   - Logical consequence chains between chunks
   - Text-based reasoning from multiple sources
   - Plausible outcomes across document