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
    pipeline.index_pdf(PDF_PATH)

    print("\nIndexing OKF concepts...")
    pipeline.index_okf_directory(OKF_PATH)

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