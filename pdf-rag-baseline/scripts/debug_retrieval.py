
from src.pipeline import PDFRAGPipeline

pipeline = PDFRAGPipeline()

queries = [
    "What activity does the textbook suggest regarding Banihal and Baramulla?",
    "Locate Banihal and Baramulla on the map of India.",
    "Railway line has been extended from Banihal to Baramulla in the Kashmir Valley.",
    "Banihal Baramula Kashmir Valley railway line activity locate towns map India",
]

for query in queries:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")

    query_vector = pipeline.embedder.embed_query(query)

    results = pipeline.vector_store.search(
        query_vector,
        top_k=10,
    )

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Document: {result.document_name}")
        print(f"Pages: {result.page_numbers}")
        print(f"Distance: {result.distance:.4f}")
        print(f"Text:\n{result.text}")
