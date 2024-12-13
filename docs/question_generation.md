# Question Generation in Yourbench

This document explains how Yourbench generates questions using language models and prompt templates.

## Overview

Yourbench uses a sophisticated prompt-based approach to generate high-quality questions from document chunks. The system supports two types of question generation:

1. Single-shot questions: Based on individual document chunks
2. Multi-hop questions: Based on pairs of related chunks

## Prompt Structure

### System Prompts

The system prompts (`fast_single_shot_system.txt` and `fast_multi_hop_system.txt`) define:

1. The role of the model as an expert educational content creator
2. Input structure and expected format
3. Analysis guidelines
4. Question types and quality requirements
5. Output format specifications

### User Prompts

The user prompts (`fast_single_shot_user.txt` and `fast_multi_hop_user.txt`) provide:

1. Document title
2. Document summary
3. Text chunk(s)
4. Target test audience

## Question Generation Process

### 1. Document Analysis

Before generating questions, the model:
- Analyzes the document content
- Identifies key concepts and themes
- Considers the test audience level
- For multi-hop: Analyzes relationships between chunks

### 2. Question Types

The system supports 10 question types:

1. **Analytical**
   - Breaks down complex ideas
   - Examines relationships between concepts

2. **Application-based**
   - Applies concepts to new scenarios
   - Tests practical understanding

3. **Clarification**
   - Seeks deeper understanding
   - Explores specific points in detail

4. **Counterfactual**
   - Explores "what if" scenarios
   - Tests understanding of causality

5. **Conceptual**
   - Examines key terms and theories
   - Tests theoretical understanding

6. **True-false**
   - Verifies basic understanding
   - Tests fact recognition

7. **Factual**
   - Tests recall of explicit information
   - Based on direct content

8. **Open-ended**
   - Encourages broader discussion
   - Tests synthesis of information

9. **False-premise**
   - Corrects misconceptions
   - Tests critical thinking

10. **Edge-case**
    - Tests boundary conditions
    - Explores limits of concepts

### 3. Difficulty Calibration

Difficulty is calibrated based on the test audience:

- Scale: 1-10
- Examples:
  - PhD level: 1 = advanced undergraduate, 10 = cutting-edge research
  - Elementary: 1 = basic recall, 10 = advanced critical thinking for age

### 4. Output Format

Questions are generated in a structured format:

```python
class QuestionAnswerPair:
    thought_process: str
    question_type: str
    question: str
    answer: str
    estimated_difficulty: int
    citations: List[str]
```

- `thought_process`: Explains the reasoning behind the question
- `question_type`: One of the 10 supported types
- `question`: The actual question text
- `answer`: Comprehensive answer
- `estimated_difficulty`: 1-10 scale
- `citations`: Direct quotes from source text

## Quality Requirements

### 1. Clarity and Precision
- Questions must be unambiguous
- All necessary context included
- No unstated assumptions

### 2. Educational Value
- Clear learning objectives
- Demonstrates understanding
- Supported by text citations

### 3. Natural Language
- Conversational tone
- Age-appropriate language
- Realistic interview style

## Implementation Details

### Single-shot Questions
1. Processes one chunk at a time
2. Focuses on content within the chunk
3. Generated through `create_single_shot_questions` pipeline

### Multi-hop Questions
1. Processes pairs of related chunks
2. Requires synthesis across chunks
3. Generated through `create_multihop_questions` pipeline

### Parallel Processing
- Uses async processing for efficiency
- Configurable concurrency limits
- Handles API rate limiting

## Best Practices

1. **Test Audience Consideration**
   - Match complexity to audience level
   - Use appropriate vocabulary
   - Scale difficulty appropriately

2. **Citation Usage**
   - Use exact quotes only
   - Support all answers with citations
   - Maintain traceability

3. **Question Diversity**
   - Mix question types
   - Vary difficulty levels
   - Cover different aspects of content

4. **Quality Control**
   - Validate JSON output
   - Check citation accuracy
   - Ensure question clarity