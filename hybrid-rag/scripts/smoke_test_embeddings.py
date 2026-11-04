from pathlib import Path

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.okf_loader import load_okf_directory
from src.chunking.text_chunker import chunk_pages
from src.embeddings.embedder import Embedder


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_DIR = PROJECT_ROOT / "data/input/sample"
OKF_DIR = PROJECT_ROOT / "data/input/unified_okf"


def main():
    print("Loading PDF...")
    pdf_files = sorted(PDF_DIR.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")

    pages_by_pdf = [
        load_pdf(pdf_path)
        for pdf_path in pdf_files
    ]

    pages = [
        page
        for pdf_pages in pages_by_pdf
        for page in pdf_pages
    ]

    chunks_by_pdf = [
        chunk_pages(pdf_pages)
        for pdf_pages in pages_by_pdf
    ]

    chunks = [
        chunk
        for pdf_chunks in chunks_by_pdf
        for chunk in pdf_chunks
    ]

    print(f"PDF files: {len(pdf_files)}")
    print(f"PDF pages: {len(pages)}")
    print(f"PDF chunks: {len(chunks)}")

    print("\nLoading OKF concepts...")
    concepts = load_okf_directory(OKF_DIR)

    print(f"OKF concepts: {len(concepts)}")

    print("\nInitialising embedding model...")
    embedder = Embedder()

    print("\nEmbedding PDF chunks...")
    pdf_embeddings = embedder.embed_pdf_chunks(chunks)

    print("Embedding OKF concepts...")
    okf_embeddings = embedder.embed_okf_concepts(concepts)

    print("\nEmbedding sample query...")
    query_embedding = embedder.embed_query(
        "What is resource planning in India?"
    )

    assert chunks, "No PDF chunks were generated"
    assert concepts, "No OKF concepts were loaded"

    pdf_dimension = len(pdf_embeddings[0].embedding)
    okf_dimension = len(okf_embeddings[0].embedding)
    query_dimension = len(query_embedding)

    print("\n--- Embedding Results ---")
    print(f"PDF embedding dimension: {pdf_dimension}")
    print(f"OKF embedding dimension: {okf_dimension}")
    print(f"Query embedding dimension: {query_dimension}")

    assert pdf_dimension == okf_dimension == query_dimension, (
        "Embedding dimensions do not match"
    )

    assert len(pdf_embeddings) == len(chunks)
    assert len(okf_embeddings) == len(concepts)

    print("\nAll embedding checks passed.")


if __name__ == "__main__":
    main()
