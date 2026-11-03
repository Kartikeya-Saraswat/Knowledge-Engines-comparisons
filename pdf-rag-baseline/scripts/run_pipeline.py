"""Run the conventional PDF RAG pipeline against the sample corpus."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import PDFRAGPipeline

PDF_DIRECTORY = PROJECT_ROOT / "data" / "input" / "sample"


def main():
    pdf_files = sorted(PDF_DIRECTORY.rglob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {PDF_DIRECTORY}")
        return

    print(f"Found {len(pdf_files)} PDF files.")

    pipeline = PDFRAGPipeline()

    # Index all PDFs
    total_chunks = 0

    for pdf_path in pdf_files:
        print(f"\nIndexing: {pdf_path.name}")

        indexed_count = pipeline.index_pdf(pdf_path)
        total_chunks += indexed_count

        print(f"Chunks indexed: {indexed_count}")

    print(f"\nTotal chunks indexed: {total_chunks}")

    # Interactive question-answering
    print("\nPDF RAG is ready. Type 'exit' to quit.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        result = pipeline.answer_question(question)

        print("\nAnswer:")
        print(result.answer)

        print("\nRetrieved sources:")
        for chunk in result.retrieved_chunks:
            print(f"\nDocument: {chunk.document_name}")
            print(f"Pages: {chunk.page_numbers}")
            print(f"Distance: {chunk.distance:.4f}")
            print(f"Chunk ID: {chunk.chunk_id}")
            print(f"Text:\n{chunk.text}")
            print("-" * 80)


if __name__ == "__main__":
    main()
