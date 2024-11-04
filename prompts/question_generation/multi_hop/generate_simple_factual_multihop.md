You will receive multiple extracts from a document along with a summary of the document. Your task is to generate factual questions that require integrating information across multiple chunks.

First, read the summary to understand the document's context, then examine all the provided text chunks.

# Guidelines
- Questions should require integration of information from MULTIPLE chunks to answer fully
- The answer must be verifiable using only the provided chunks
- Focus on factual, verifiable information rather than opinions
- Vary the difficulty level of the questions from 1 (very easy) to 5 (very challenging)
- Provide clear reasoning for how the question tests integration of knowledge across chunks

# Task Steps
1. **Read the Summary:** Understand the general context and purpose of the document.
2. **Analyze All Chunks:** Examine all provided chunks, looking for connections and relationships between information.
3. **Map Information:** Create a mental map of how facts from different chunks relate to each other.
4. **Identify Integration Points:** Look for topics that span multiple chunks and require synthesis.
5. **Design Integrative Questions:** Create questions that cannot be answered from a single chunk alone.
6. **Identify Direct Quotes:** Gather exact, VERBATIM lines from multiple chunks that support the questions and answers.
7. **Create Question-Answer Pairs:** Formulate the questions and provide comprehensive answers that show integration.
8. **Assign Difficulty Level:** Determine difficulty level (1-5) based on:
   - Number of chunks needed for the answer
   - Complexity of relationships between facts
   - Level of synthesis required

# Output Format

Output the questions as an array of JSON objects according to the specified Pydantic structure. Your generated questions should be in the <generated_questions> tags, and will be validated externally against the following Pydantic structure:

```python
class GeneratedIntegrativeQAPair(BaseModel):
    document_analysis: str = Field(..., description="The analysis of the document, given its summary.")
    chunks_analysis: List[ChunkAnalysis] = Field(..., description="Analysis of each provided chunk and its relationship to others")
    integration_points: List[str] = Field(..., description="Key points where information from multiple chunks connects or relates")
    potential_integrative_questions: List[str] = Field(..., description="Possible questions requiring synthesis across chunks")
    best_direction: str = Field(..., description="The chosen question direction and why it effectively tests integration of knowledge")
    direct_line_quotes: Dict[str, List[str]] = Field(..., description="Dictionary mapping chunk IDs to verbatim quotes used to answer the question")
    question: str = Field(..., description="The integrative question")
    answer: str = Field(..., description="The answer showing synthesis across chunks")
    reasoning: str = Field(..., description="How the answer integrates information from multiple chunks")
    chunks_used: List[str] = Field(..., description="List of chunk IDs used to answer the question")
    kind: str = Field(..., description="Type of question (e.g., 'integrative_factual', 'cross_reference')")
    estimated_difficulty: int = Field(..., description="Difficulty (1-5) based on integration complexity")

class ChunkAnalysis(BaseModel):
    chunk_id: str
    content_summary: str
    relationships: List[str] = Field(..., description="How this chunk relates to others")
```

# Example Output

<generated_questions>
[
    {
        "document_analysis": "The document examines the progression of semiconductor technology from its early inception in the 1950s through its transformation in the 1980s, focusing on pivotal technical milestones and their implications for industry applications.",
        "chunks_analysis": [
            {
                "chunk_id": "chunk_1",
                "content_summary": "Discusses the initial development of semiconductors and silicon-based transistors in the 1950s, emphasizing foundational innovations and early technical constraints.",
                "relationships": ["Sets the groundwork for subsequent developments in chunk_2", "Introduces material challenges explored further in chunk_3"]
            },
            {
                "chunk_id": "chunk_2",
                "content_summary": "Outlines major advancements in semiconductor fabrication during the 1970s, highlighting the shift to large-scale integration (LSI) and the advent of microprocessors.",
                "relationships": ["Builds upon foundational work from chunk_1", "Enables commercial computer applications explored in chunk_4"]
            },
            {
                "chunk_id": "chunk_3",
                "content_summary": "Explores the material challenges faced during the transition from vacuum tubes to transistors, focusing on reliability and power efficiency improvements.",
                "relationships": ["Delves into the material limitations first mentioned in chunk_1", "Contributes to the advancements detailed in chunk_2"]
            }
        ],
        "integration_points": [
            "Understanding of the semiconductor evolution from basic transistor design to highly integrated circuits",
            "Linking improvements in manufacturing processes with broad commercial and industrial applications"
        ],
        "potential_integrative_questions": [
            "How did innovations in semiconductor manufacturing contribute to the viability of early computer systems?",
            "What material and design breakthroughs were necessary to progress from basic transistors to complex microprocessors?"
        ],
        "best_direction": "A chronological analysis tracing semiconductor advancements alongside their commercial impacts illustrates the interplay between technical evolution and market demands.",
        "direct_line_quotes": {
            "chunk_1": [
                "Silicon-based transistors were limited by thermal instability and required precise manufacturing conditions",
                "Initial transistor designs struggled with reliability and short operational lifespans"
            ],
            "chunk_2": [
                "The advent of large-scale integration (LSI) in the 1970s allowed for microprocessors with thousands of transistors, significantly enhancing computational power",
                "These improvements set the stage for commercial computing applications, from business data processing to consumer electronics"
            ],
            "chunk_3": [
                "Material challenges, such as optimizing silicon purity, were critical to improving reliability and efficiency",
                "Advances in semiconductor materials marked a turning point in the practicality of these components for high-frequency applications"
            ]
        },
        "question": "How did the evolution from single-transistor designs to large-scale integration impact both the performance capabilities and the commercial feasibility of semiconductor-based devices?",
        "answer": "The shift from single-transistor designs to large-scale integration (LSI) revolutionized semiconductor devices by vastly improving processing power, reducing production costs, and increasing reliability. These advances enabled the development of microprocessors, catalyzing the growth of the personal computer industry and making digital devices more accessible for commercial and consumer markets.",
        "reasoning": "Examining the transition from single-transistor designs to LSI illustrates how overcoming early material and thermal constraints allowed for exponential improvements in efficiency and reliability, paving the way for commercial computing applications. This shift highlights the close link between technical advancements and market expansion.",
        "chunks_used": ["chunk_1", "chunk_2", "chunk_3"],
        "kind": "integrative_factual",
        "estimated_difficulty": 5
    }
]
</generated_questions>

# Notes
- Questions must require information from multiple chunks to answer fully
- Ensure proper cross-referencing between chunks in your analysis
- The JSON output must pass Pydantic validation
- Use <scratchpad> tags for your analysis notes
- The answer must be fully verifiable using only the provided chunks
- Identify and explain relationships between information in different chunks
- Use <generated_questions> tags for the output

This prompt revision focuses on generating questions that test understanding across multiple chunks of text, requiring synthesis and integration of information. The Pydantic structure has been expanded to track relationships between chunks and ensure proper documentation of how information is integrated in the answers.