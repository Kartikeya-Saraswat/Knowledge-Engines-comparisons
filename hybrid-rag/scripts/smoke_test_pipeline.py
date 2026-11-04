from pathlib import Path

from src.pipeline import HybridRAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = (
    PROJECT_ROOT
    / "../pdf-to-okf-auto/data/input/sample/jess101.pdf"
).resolve()

OKF_PATH = (
    PROJECT_ROOT
    / "../pdf-to-okf-auto/data/output/pilot_jess101/unified"
).resolve()


def main():
    pipeline = HybridRAGPipeline()

    print("Indexing PDF...")
    pdf_count = pipeline.index_pdf(PDF_PATH)
    print(f"PDF records indexed/updated: {pdf_count}")

    print("\nIndexing OKF concepts...")
    okf_count = pipeline.index_okf_directory(OKF_PATH)
    print(f"OKF records indexed/updated: {okf_count}")

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