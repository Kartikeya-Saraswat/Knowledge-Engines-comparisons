from src.generation.answer_generator import AnswerGenerator
from src.retrieval.vector_store import RetrievedChunk

question = "What activity does the textbook suggest regarding Banihal and Baramula?"

relevant_chunk = RetrievedChunk(
    chunk_id="generation-test-001",
    text=(
        "Activity Railway line has been extended from Banihal to "
        "Baramula in the Kashmir Valley. Locate these two towns "
        "on the map of India."
    ),
    document_name="jess107.pdf",
    page_numbers=(5, 6),
    distance=0.0,
)

generator = AnswerGenerator()

answer = generator.generate(question, [relevant_chunk])

print("\nQUESTION:")
print(question)

print("\nGENERATED ANSWER:")
print(answer)