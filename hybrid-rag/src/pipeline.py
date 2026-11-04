from pathlib import Path

from src.chunking.text_chunker import chunk_pages
from src.embeddings.embedder import Embedder
from src.fusion.ranker import reciprocal_rank_fusion, FusedRecord
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.okf_loader import load_okf_directory
from src.retrieval.vector_store import VectorStore, RetrievedRecord


class HybridRAGPipeline:
    def __init__(
        self,
        persist_directory: str | Path = "data/chroma_db",
        embedder: Embedder | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = VectorStore(persist_directory)

    def index_pdf(self, pdf_path: str | Path) -> int:
        """Load, chunk, embed and index a PDF."""

        pages = load_pdf(pdf_path)
        chunks = chunk_pages(pages)
        embedded_records = self.embedder.embed_pdf_chunks(chunks)

        return self.vector_store.upsert_pdf_chunks(embedded_records)

    def index_okf_directory(self, directory: str | Path) -> int:
        """Load, embed and index all OKF concepts in a directory."""

        concepts = load_okf_directory(directory)
        embedded_records = self.embedder.embed_okf_concepts(concepts)

        return self.vector_store.upsert_okf_concepts(embedded_records)

    def retrieve_pdf(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedRecord]:
        """Retrieve relevant PDF chunks."""

        query_embedding = self.embedder.embed_query(query)

        return self.vector_store.search_pdf(
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def retrieve_okf(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedRecord]:
        """Retrieve relevant OKF concepts."""

        query_embedding = self.embedder.embed_query(query)

        return self.vector_store.search_okf(
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def retrieve_hybrid(
        self,
        query: str,
        pdf_top_k: int = 5,
        okf_top_k: int = 5,
        final_top_k: int = 5,
    ) -> list[FusedRecord]:
        """Retrieve from PDF and OKF, then combine using RRF."""

        if final_top_k <= 0:
            raise ValueError("final_top_k must be greater than zero")

        query_embedding = self.embedder.embed_query(query)

        pdf_results = self.vector_store.search_pdf(
            query_embedding=query_embedding,
            top_k=pdf_top_k,
        )

        okf_results = self.vector_store.search_okf(
            query_embedding=query_embedding,
            top_k=okf_top_k,
        )

        fused_results = reciprocal_rank_fusion(
            pdf_results=pdf_results,
            okf_results=okf_results,
        )

        return fused_results[:final_top_k]

    def collection_counts(self) -> dict[str, int]:
        return self.vector_store.collection_counts()