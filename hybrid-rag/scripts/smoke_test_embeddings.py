from pathlib import Path

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.okf_loader import load_okf_directory
from src.chunking.text_chunker import chunk_pages
from src.embeddings.embedder import Embedder


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
    print("Loading PDF...")
    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages)

    print(f"PDF pages: {len(pages)}")
    print(f"PDF chunks: {len(chunks)}")

    print("\nLoading OKF concepts...")
    concepts = load_okf_directory(OKF_PATH)

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