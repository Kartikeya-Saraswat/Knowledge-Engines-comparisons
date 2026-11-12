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

    question = "What is resource planning in India?"

    print(f"\nQuestion: {question}")
    print("\nGenerating grounded answer using Ollama...")

    result = pipeline.answer_question(
        question=question,
        pdf_top_k=3,
        okf_top_k=3,
        final_top_k=5,
    )

    evidence = pipeline.retrieve_hybrid(
        question,
        pdf_top_k=3,
        okf_top_k=3,
        final_top_k=5,
    )

    print("\n--- Fused Evidence ---")

    for index, item in enumerate(evidence, start=1):
        print(f"\n[E{index}]")
        print("ID:", item.record_id)
        print("Source:", item.metadata["source_type"])
        print("RRF score:", item.rrf_score)
        print("Source ranks:", item.source_ranks)
        print("Full text:", item.text)
    
    print("\n--- Generated Answer ---")
    print(result.answer)

    print("\n--- Source Citations ---")

    for citation in result.citations:
        print(f"\nEvidence ID: {citation.evidence_id}")
        print(f"Source type: {citation.source_type}")
        print(f"Record ID: {citation.record_id}")
        print(f"Source: {citation.description}")

    print("\nAnswer generation smoke test completed.")


if __name__ == "__main__":
    main()

    