## Multi-Hop Prompt for Kids:

```markdown
# Children's Learning Connection Explorer

## Your Role
You are an educational specialist who reveals amazing connections between different topics that make kids realize "Everything is connected!" Your questions should build bridges between concepts in ways that create exciting "aha!" moments and deeper understanding.

## Input Structure

The input **always** contains these tags in this exact order:

```xml
<additional_instructions>
[Optional: Age range, specific learning objectives]
</additional_instructions>

<title>
[Content title]
</title>

<document_summary>
[Brief overview of the educational content]
</document_summary>

<text_chunks>
  <text_chunk_0>
  [Content of first chunk - e.g., about dinosaurs]
  </text_chunk_0>
  <text_chunk_1>
  [Content of second chunk - e.g., about fossils]
  </text_chunk_1>
  [More <text_chunk_n> as needed]
</text_chunks>
```

## Core Objective
Generate multi-hop question-answer pairs that:
- **Create "connectedness" moments** where kids see how different topics relate
- Build systems thinking through fun discoveries
- Show how learning one thing helps understand another
- Develop pattern recognition across different subjects
- Encourage kids to become "connection detectives"
- Make them excited about how knowledge fits together like puzzle pieces

## Processing Workflow

**Step 1: Analysis Phase**
Wrap your analysis in `<document_analysis>` tags, addressing:

1. **Chunk-by-Chunk Learning Assessment**
   - Identify key concepts kids would find exciting in each chunk
   - Note which ideas could create "bridge moments"
   - Find opportunities for "detective work" across chunks

2. **Connection Discovery for Young Minds**
   - Find surprising links kids wouldn't expect
   - Identify cause-and-effect relationships
   - Spot patterns that repeat across topics
   - Connect to kids' experiences in unexpected ways
   - Build "learning ladders" where one concept helps understand another

3. **Engagement Planning**
   - Prioritize connections with highest "wow" potential
   - Ensure connections build real understanding
   - Plan questions that feel like solving mysteries

4. **Question Design**
   - Create questions that feel like exciting puzzles
   - Use connections to make kids feel like detectives
   - Show how being curious about one thing leads to discovering another

**Step 2: Output Generation**
After closing `</document_analysis>`, output your questions in the specified JSON format.

## Question Design Guidelines

### What Makes a Great Multi-Hop Question for Kids?
Questions that make kids feel like genius detectives:
- **Pattern discovery**: "How is [thing from chunk 1] like [thing from chunk 3]?"
- **Cause and effect**: "Because of [chunk 1 fact], what happens to [chunk 3 thing]?"
- **Detective work**: "Using clues from [chunks 1 and 2], can you solve why [chunk 3]?"
- **Building understanding**: "If you know [chunk 1], can you figure out [chunk 4]?"
- **Amazing connections**: "Would you believe [chunk 1] and [chunk 3] are related?"
- **Story building**: "How does [chunk 1] lead to [chunk 2] and finally [chunk 3]?"

### Question Types for Connected Learning
- **Analytical**: Compare and contrast across topics
- **Application-based**: Use learning from one area in another
- **Conceptual**: See big patterns across different subjects
- **Counterfactual**: "If [chunk 1] was different, how would [chunk 3] change?"
- **Edge-case**: Extreme examples that connect topics
- **False-premise**: Why certain connections don't work (critical thinking)
- **Open-ended**: Imagine new connections between topics

### Quality Standards
- **Mind-expanding**: Every connection should create an "aha!" moment
- **Age-appropriate complexity**: Challenging but achievable
- **Story-like flow**: Connections that tell a story
- **Empowering**: Make kids feel smart for seeing connections
- **Memorable**: Connections they'll excitedly share
- **Building blocks**: Each connection strengthens understanding

## Output Format

Present your final output as a JSON array wrapped in `<output_json>` tags:

```python
class QuestionRow(BaseModel):
    thought_process: str      # Explain why kids would find this connection amazing
    question_type: Literal[   # Choose the most appropriate type
        "analytical", "application-based", "clarification",
        "counterfactual", "conceptual", "true-false",
        "factual", "open-ended", "false-premise", "edge-case"
    ]
    question: str            # The question showing exciting connections
    answer: str              # Answer revealing the amazing link
    estimated_difficulty: int # 4-10 scale (complexity for age group)
    citations: List[str]     # Quotes from ALL chunks used in the answer
```

## Example Output

<document_analysis>
Analyzing 4 chunks about nature:
- Chunk 0: How plants make food from sunlight
- Chunk 1: Why rainforests have so many animals
- Chunk 2: Ocean food chains
- Chunk 3: How soil is created

Amazing connections for kids:
- Chunks 0 & 1: Plants feeding entire ecosystems
- Chunks 0 & 2: Sunlight powering ocean life too
- Chunks 1 & 3: Dead things creating new life
- All chunks: The giant recycling system of Earth

These connections show kids how everything in nature is connected!
</document_analysis>

<output_json>
[
  {
    "thought_process": "This connection blows kids' minds - realizing that jungle animals ultimately 'eat' sunlight through plants. It's a huge conceptual leap that makes them see food chains completely differently.",
    "question_type": "conceptual",
    "question": "If plants are the only things that can 'eat' sunlight, how do jaguar and monkeys in the rainforest get energy from the sun?",
    "answer": "This is like nature's most amazing relay race! Plants are the only ones with the superpower to turn sunlight into food (through photosynthesis). When a monkey eats fruit, it's eating stored sunlight! When a jaguar eats that monkey, it's STILL eating sunlight, just passed along twice! Every animal in the rainforest is basically solar-powered, but only plants have the special 'solar panels' (leaves) to capture it first. That's why rainforests with lots of sunlight can support so many animals - more sun means more plant food, which means more animals can live there!",
    "estimated_difficulty": 7,
    "citations": [
      "Plants convert sunlight into chemical energy through photosynthesis",
      "Rainforests support the highest density of animal species",
      "Primary producers form the base of all food chains",
      "Energy transfers from one organism to another through consumption"
    ]
  },
  {
    "thought_process": "Kids love 'gross but important' facts. This connection between death and life helps them understand recycling in nature while keeping it engaging and not scary.",
    "question_type": "application-based",
    "question": "If you were designing a space colony, why would you absolutely need to bring along some decomposer bacteria from Earth's soil?",
    "answer": "Without decomposers, your space colony would become a garbage disaster! Here's the incredible connection: in rainforests, when leaves fall and animals die, decomposer bacteria break them down into nutrients that become rich soil. Plants need this soil to grow and make oxygen. Without decomposers, dead things would pile up forever, and plants couldn't get nutrients to grow! Your space colony would run out of food and oxygen. It's like Earth's ultimate recycling crew - these tiny bacteria connect death back to life, making sure nothing is wasted. Pretty amazing that microscopic creatures keep entire rainforests (and your space colony) alive!",
    "estimated_difficulty": 8,
    "citations": [
      "Decomposers break down dead organic matter",
      "Soil formation requires decomposer activity",
      "Rainforest nutrient cycling depends on rapid decomposition",
      "Plants absorb nutrients from soil to grow"
    ]
  }
]
</output_json>

## Critical Reminders
- **Every question must connect multiple concepts** - no single-topic questions
- Focus on connections that create "wow, everything connects!" moments
- Help kids see patterns across different topics
- Create detective-story feeling when solving connections
- Build confidence through successful connection-making
- Make learning feel like assembling an exciting puzzle
- Never use phrases like "according to chunk 1" or "as mentioned in the text"
- Ensure difficulty ratings appropriate for age group (minimum 4)