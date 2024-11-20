# Multi-Document Comprehension Question Generation

You will receive multiple document extracts from the same source document and a summary. Your task is to generate high-quality true-false questions that test comprehension across these extracts, requiring understanding of relationships and connections between different parts of the text.

## Core Principles

1. **Multi-Document Evidence**
   - Questions MUST be answerable from the provided text extracts
   - Required verbatim quotes from multiple chunks where applicable
   - No external knowledge requirements
   - No inference beyond what's explicitly stated
   - Clear connections between chunks must be established

2. **Question Diversity**
   - Mix of single-chunk and multi-chunk questions
   - Varied difficulty levels (1-5)
   - Different types of comprehension (main ideas, details, relationships)
   - Balance of true and false statements
   - Focus on relationships between information in different chunks

3. **Question Quality**
   - Clear, unambiguous language
   - No trick questions or wordplay
   - Realistic and meaningful assessments
   - Explicit connection points between chunks

## Data Model

Here is the pydantic model for the output. You must generate valid JSONs that match this output format, including all the fields. Your responses will be validated against this model, and you will be penalized if any of the fields are missing or invalid.

```python
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, constr

class QuestionType(str, Enum):
    TRUE_FALSE = "true-false"

class DifficultyLevel(int, Enum):
    VERY_EASY = 1  # Surface-level fact recognition
    EASY = 2       # Basic comprehension
    MEDIUM = 3     # Relationship understanding
    HARD = 4       # Complex relationships
    VERY_HARD = 5  # Nuanced understanding

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
    text_based: bool = Field(..., description="Answerable from text alone")
    no_tricks: bool = Field(..., description="Avoids misleading wordplay")

class ChunkAnalysis(BaseModel):
    chunk_id: str
    content_summary: str
    relevant_information: str
    connection_points: List[str]

class GeneratedQuestionAnswerPair(BaseModel):
    """
    Represents a structured QA pair for multi-document comprehension testing.
    """
    # Analysis Fields
    document_extract_analysis: str = Field(
        ...,
        description="Analysis of the key points and structure of the extract",
        min_length=50
    )
    
    chunk_analyses: List[ChunkAnalysis] = Field(..., min_items=2)
    
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
        min_length=10
    )

    # Core Question Fields
    kind: QuestionType = Field(
        default=QuestionType.TRUE_FALSE,
        description="Question type (true-false)"
    )
    
    question: constr(min_length=10) = Field(
        ...,
        description="The true-false statement to evaluate"
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

1. **Multi-Chunk Analysis**
   - Read and understand each chunk carefully
   - Identify connections between chunks
   - Map relationships and dependencies
   - Note potential areas for cross-chunk testing

2. **Concept Identification**
   - List main ideas that span multiple chunks
   - Identify important details that connect chunks
   - Note relationships between concepts across chunks
   - Look for cause-effect relationships that bridge chunks

3. **Question Formation**
   - Create diverse questions requiring multiple chunk comprehension
   - Ensure direct text evidence exists from relevant chunks
   - Vary difficulty levels based on number of chunks needed
   - Maintain balance of true/false answers

4. **Quality Verification**
   - Check each question against quality metrics
   - Verify supporting quotes are verbatim from appropriate chunks
   - Ensure no external knowledge needed
   - Confirm clear, unambiguous language
   - Validate chunk connections

## Examples

### Example 1: Technological Innovation Impact Analysis

```json
{
    "document_extract_analysis": "The extracts explore how quantum computing developments affect cryptography and cybersecurity across different sectors, with detailed analysis of both current implementations and future implications.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Quantum computing basics and current state",
            "relevant_information": "Current quantum computers achieve 100 qubit processing",
            "connection_points": ["quantum technology", "processing power", "implementation timeline"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Cryptographic implications",
            "relevant_information": "Traditional encryption methods vulnerable to quantum attacks",
            "connection_points": ["security impact", "technological threats", "adaptation needs"]
        }
    ],
    "testable_concepts": [
        "quantum computing capabilities",
        "cryptographic vulnerability",
        "technological adaptation",
        "security implications"
    ],
    "potential_question_directions": [
        "How does quantum computing advancement affect current security?",
        "What is the relationship between qubit processing and encryption?",
        "How must cybersecurity evolve in response to quantum threats?"
    ],
    "best_direction": "How does quantum computing advancement affect current security?",
    "comprehension_type": "implication_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Current quantum computers achieve 100 qubit processing",
        "Traditional encryption methods vulnerable to quantum attacks"
    ],
    "quote_context": "The quotes establish the current state of quantum computing and its direct impact on existing security measures",
    "kind": "true-false",
    "question": "The achievement of 100 qubit processing in quantum computers has already rendered all traditional encryption methods obsolete.",
    "answer": "False",
    "reasoning": "While the text indicates that traditional encryption methods are vulnerable to quantum attacks, it doesn't state that current 100 qubit quantum computers have already rendered these methods obsolete. The text suggests vulnerability rather than current obsolescence.",
    "difficulty": 4,
    "difficulty_justification": "Requires careful analysis of the relationship between current quantum capabilities and their actual impact on encryption, distinguishing between vulnerability and obsolescence."
}
```

### Example 2: Environmental Policy Implementation

```json
{
    "document_extract_analysis": "The extracts detail various carbon pricing mechanisms and their implementation across different regions, including economic impacts and effectiveness metrics.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Carbon pricing mechanisms and theory",
            "relevant_information": "Carbon tax vs cap-and-trade implementation",
            "connection_points": ["policy design", "economic tools", "implementation methods"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Regional implementation results",
            "relevant_information": "European carbon market reduced emissions by 23%",
            "connection_points": ["effectiveness metrics", "regional variation", "outcome measurement"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Economic impact analysis",
            "relevant_information": "GDP impact varies by implementation method",
            "connection_points": ["economic effects", "policy outcomes", "measurement metrics"]
        }
    ],
    "testable_concepts": [
        "carbon pricing mechanisms",
        "implementation effectiveness",
        "economic impacts",
        "regional variations"
    ],
    "potential_question_directions": [
        "How do different carbon pricing mechanisms compare in effectiveness?",
        "What determines the economic impact of carbon pricing?",
        "How do regional implementations affect outcomes?"
    ],
    "best_direction": "How do different carbon pricing mechanisms compare in effectiveness?",
    "comprehension_type": "compare_contrast",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Carbon tax vs cap-and-trade implementation",
        "European carbon market reduced emissions by 23%",
        "GDP impact varies by implementation method"
    ],
    "quote_context": "The quotes connect different pricing mechanisms to their implementation outcomes and economic effects",
    "kind": "true-false",
    "question": "The European carbon market's 23% emissions reduction proves that cap-and-trade systems are universally more effective than carbon taxes.",
    "answer": "False",
    "reasoning": "While the text shows the European carbon market's success with a 23% reduction, it doesn't compare this directly with carbon tax effectiveness or make universal claims about cap-and-trade superiority. The text actually indicates that effectiveness varies by implementation method.",
    "difficulty": 5,
    "difficulty_justification": "Requires synthesizing information about different pricing mechanisms, understanding the limitations of regional results, and avoiding overgeneralization from specific examples."
}
```

### Example 3: Medical Research Development

```json
{
    "document_extract_analysis": "The extracts cover the development process of mRNA vaccines, from initial research through clinical trials to mass production challenges.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Basic mRNA research history",
            "relevant_information": "30 years of foundational research preceded COVID vaccines",
            "connection_points": ["research timeline", "scientific foundation", "development process"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Clinical trial acceleration",
            "relevant_information": "Parallel trial phases during pandemic",
            "connection_points": ["testing process", "timeline compression", "safety protocols"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Production scaling",
            "relevant_information": "New manufacturing facilities built in 6 months",
            "connection_points": ["manufacturing challenges", "timeline pressure", "resource allocation"]
        }
    ],
    "testable_concepts": [
        "research development timeline",
        "clinical trial process",
        "manufacturing scale-up",
        "emergency adaptation"
    ],
    "potential_question_directions": [
        "How did prior research enable rapid vaccine development?",
        "What allowed for accelerated clinical trials?",
        "How was manufacturing capacity rapidly expanded?"
    ],
    "best_direction": "How did prior research enable rapid vaccine development?",
    "comprehension_type": "process_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "30 years of foundational research preceded COVID vaccines",
        "Parallel trial phases during pandemic",
        "New manufacturing facilities built in 6 months"
    ],
    "quote_context": "The quotes establish the timeline of development from research through trials to production",
    "kind": "true-false",
    "question": "The rapid development of COVID mRNA vaccines was achieved without any prior research foundation or existing scientific knowledge.",
    "answer": "False",
    "reasoning": "The text explicitly states that 30 years of foundational research preceded COVID vaccines, indicating that the rapid development was built on extensive prior scientific work rather than starting from scratch.",
    "difficulty": 3,
    "difficulty_justification": "Requires connecting information about research history with development timeline, but the relationship is fairly directly stated in the text."
}
```

### Example 4: Artificial Intelligence Ethics

```json
{
    "document_extract_analysis": "The extracts examine ethical considerations in AI development, focusing on bias detection, accountability mechanisms, and regulatory frameworks.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "AI bias detection methods",
            "relevant_information": "Automated bias detection systems identify 78% of cases",
            "connection_points": ["technical methods", "effectiveness metrics", "implementation challenges"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Regulatory framework development",
            "relevant_information": "Five-tier accountability system proposed",
            "connection_points": ["governance structure", "responsibility levels", "implementation requirements"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Corporate implementation challenges",
            "relevant_information": "42% of companies struggle with compliance",
            "connection_points": ["practical challenges", "resource requirements", "adaptation difficulties"]
        }
    ],
    "testable_concepts": [
        "bias detection effectiveness",
        "regulatory compliance",
        "implementation challenges",
        "accountability systems"
    ],
    "potential_question_directions": [
        "How effective are current bias detection methods?",
        "What makes regulatory compliance challenging?",
        "How do accountability systems address bias issues?"
    ],
    "best_direction": "What makes regulatory compliance challenging?",
    "comprehension_type": "system_impact",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Automated bias detection systems identify 78% of cases",
        "Five-tier accountability system proposed",
        "42% of companies struggle with compliance"
    ],
    "quote_context": "The quotes connect technical capabilities with regulatory requirements and implementation challenges",
    "kind": "true-false",
    "question": "Companies struggling with AI regulatory compliance can rely solely on automated bias detection systems to meet all accountability requirements.",
    "answer": "False",
    "reasoning": "While automated systems detect 78% of bias cases, the text indicates a more comprehensive five-tier accountability system is required, and companies face broader compliance challenges beyond just bias detection.",
    "difficulty": 4,
    "difficulty_justification": "Requires synthesizing information about technical capabilities, regulatory requirements, and implementation challenges to understand why automated systems alone are insufficient."
}
```

### Example 5: Renewable Energy Integration

```json
{
    "document_extract_analysis": "The extracts discuss the integration of renewable energy sources into existing power grids, including technical challenges, storage solutions, and grid modernization efforts.",
    "chunk_analyses": [
        {
            "chunk_id": "chunk1",
            "content_summary": "Grid stability challenges",
            "relevant_information": "Intermittent renewable sources cause 15% voltage fluctuation",
            "connection_points": ["technical challenges", "stability issues", "integration problems"]
        },
        {
            "chunk_id": "chunk2",
            "content_summary": "Storage technology development",
            "relevant_information": "New battery systems achieve 92% efficiency",
            "connection_points": ["technical solutions", "efficiency metrics", "implementation progress"]
        },
        {
            "chunk_id": "chunk3",
            "content_summary": "Smart grid implementation",
            "relevant_information": "AI-driven grid management reduces instability by 67%",
            "connection_points": ["technological solutions", "management systems", "effectiveness metrics"]
        }
    ],
    "testable_concepts": [
        "grid stability",
        "storage efficiency",
        "smart management",
        "system integration"
    ],
    "potential_question_directions": [
        "How do storage solutions affect grid stability?",
        "What role does AI play in renewable integration?",
        "How do different technologies work together in grid modernization?"
    ],
    "best_direction": "How do different technologies work together in grid modernization?",
    "comprehension_type": "system_analysis",
    "quality_metrics": {
        "clear_language": true,
        "text_based": true,
        "no_tricks": true
    },
    "supporting_quotes": [
        "Intermittent renewable sources cause 15% voltage fluctuation",
        "New battery systems achieve 92% efficiency",
        "AI-driven grid management reduces instability by 67%"
    ],
    "quote_context": "The quotes establish the relationship between problems, solutions, and management systems in grid modernization",
    "kind": "true-false",
    "question": "High-efficiency battery systems eliminate the need for AI-driven grid management in handling renewable energy integration.",
    "answer": "False",
    "reasoning": "The text shows that both technologies play complementary roles: while battery systems achieve 92% efficiency, AI-driven management provides additional stability benefits, reducing instability by 67%. Neither technology alone completely solves the integration challenges.",
    "difficulty": 5,
    "difficulty_justification": "Requires understanding the complex interplay between multiple technologies and their distinct but complementary roles in addressing grid stability challenges."
}
```


## Common Pitfalls to Avoid

1. **Single-Chunk Focus**
   ❌ Questions that can be answered from one chunk alone
   ✅ Questions that require synthesizing information across chunks

2. **Missing Connections**
   ❌ Questions that don't clearly relate information between chunks
   ✅ Questions that explicitly connect concepts across chunks

3. **Temporal Confusion**
   ❌ Questions that mix up the sequence of events from different chunks
   ✅ Questions that maintain clear chronological relationships

4. **External Knowledge**
   ❌ Questions requiring information beyond the provided chunks
   ✅ Questions answerable solely from the provided chunks

5. **Ambiguous References**
   ❌ Unclear which chunk information comes from
   ✅ Clear attribution and connection of information sources

## Output Requirements

1. Generate 3-5 questions per set of chunks
2. Ensure diverse comprehension types
3. Include at least one question requiring multiple chunk synthesis
4. Include at least one question at difficulty level 4 or 5
5. Provide verbatim supporting quotes from relevant chunks
6. Include complete chunk analyses and quality metrics for each question

## Example Output Format

Enclose your output in <generated_questions> tags:

```json
<generated_questions>
[
    {
        // Question 1 (Medium/Cross-Chunk Synthesis)
    },
    {
        // Question 2 (Hard/System Understanding)
    },
    {
        // Question 3 (Very Hard/Timeline Analysis)
    },
    // ...
]
</generated_questions>
```