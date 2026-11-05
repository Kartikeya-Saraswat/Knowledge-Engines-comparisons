import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2] / "pdf-rag-baseline"
sys.path.insert(0, str(_PROJECT_ROOT))
import json
import sys
from pathlib import Path

from src.pipeline import PDFRAGPipeline
from src.retrieval.vector_store import VectorStore


def main():
    request = json.load(sys.stdin)

    index_dir = Path(request["index_directory"]).resolve()
    store = VectorStore(persist_directory=index_dir)
    pipeline = PDFRAGPipeline(vector_store=store)

    operation = request["operation"]

    if operation == "index":
        pdf_dir = Path(request["pdf_directory"]).resolve()
        pdfs = sorted(pdf_dir.glob("*.pdf"))

        total_chunks = 0
        for pdf in pdfs:
            total_chunks += pipeline.index_pdf(pdf)

        result = {
            "status": "success",
            "indexed_files": len(pdfs),
            "indexed_chunks": total_chunks,
            "index_directory": str(index_dir),
        }

    elif operation == "retrieve":
        question = request["question"]
        top_k = request.get("top_k", 5)

        embedding = pipeline.embedder.embed_query(question)
        chunks = pipeline.vector_store.search(
            query_embedding=embedding,
            top_k=top_k,
        )

        result = {
            "status": "success",
            "results": [
                {
                    "record_id": chunk.chunk_id,
                    "text": chunk.text,
                    "document_name": chunk.document_name,
                    "page_numbers": chunk.page_numbers,
                    "distance": chunk.distance,
                    "source_type": "pdf",
                }
                for chunk in chunks
            ],
        }

    else:
        raise ValueError(f"Unknown operation: {operation}")

    print(json.dumps(result, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Worker error: {exc}", file=sys.stderr)
        raise
