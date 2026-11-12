from pathlib import Path

from src.pipeline import HybridRAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_DIR = (
    PROJECT_ROOT.parent
    / "pdf-to-okf-auto"
    / "data"
    / "input"
    / "sample"
)

OKF_DIR = (
    PROJECT_ROOT.parent
    / "pdf-to-okf-auto"
    / "data"
    / "output"
    / "ollama_full_run"
    / "unified"
)

def main():
    pipeline = HybridRAGPipeline(
        persist_directory=PROJECT_ROOT / "data" / "chroma_smoke_test"
    )

    print("Clearing and rebuilding the index from all source files...")
    counts = pipeline.rebuild_index(PDF_DIR, OKF_DIR)
    print(f"PDF files: {counts['pdf_files']}")
    print(f"PDF chunks indexed: {counts['pdf_chunks']}")
    print(f"OKF concepts indexed: {counts['okf_concepts']}")

    print("\nCollection counts:")
    for name, count in pipeline.collection_counts().items():
        print(f"{name}: {count}")

    query = "What is resource planning in India?"

    print(f"\nQuery: {query}")

    print("\n--- PDF Retrieval ---")
    pdf_results = pipeline.retrieve_pdf(query, top_k=3)

    for result in pdf_results:
        print(f"\nID: {result.record_id}")
        print(f"Document: {result.metadata['document_name']}")
        print(f"Pages: {result.metadata['page_numbers']}")
        print(f"Distance: {result.distance:.4f}")
        print(f"Text: {result.text[:300]}")

    print("\n--- OKF Retrieval ---")
    okf_results = pipeline.retrieve_okf(query, top_k=3)

    for result in okf_results:
        print(f"\nID: {result.record_id}")
        print(f"Concept: {result.metadata['title']}")
        print(f"Distance: {result.distance:.4f}")
        print(f"Text: {result.text[:300]}")

    assert pdf_results, "PDF retrieval returned no results"
    assert okf_results, "OKF retrieval returned no results"

    print("\n--- Hybrid Retrieval (RRF) ---")

    hybrid_results = pipeline.retrieve_hybrid(
        query,
        pdf_top_k=3,
        okf_top_k=3,
        final_top_k=5,
    )

    for result in hybrid_results:
        print(f"\nID: {result.record_id}")
        print(f"Source: {result.metadata['source_type']}")
        print(f"RRF score: {result.rrf_score:.6f}")
        print(f"Source ranks: {result.source_ranks}")
        print(f"Text: {result.text[:300]}")

    assert hybrid_results, "Hybrid retrieval returned no results"

    print("\nEnd-to-end indexing and retrieval checks passed.")

if __name__ == "__main__":
    main()
