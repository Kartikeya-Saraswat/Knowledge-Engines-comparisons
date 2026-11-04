from pathlib import Path

from src.chunking.text_chunker import chunk_pages
from src.embeddings.embedder import Embedder
from src.fusion.ranker import reciprocal_rank_fusion, FusedRecord
from src.generation.answer_generator import AnswerGenerator, GeneratedAnswer
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.okf_loader import load_okf_directory
from src.retrieval.vector_store import VectorStore, RetrievedRecord


class HybridRAGPipeline:
    def __init__(
        self,
        persist_directory: str | Path = "data/chroma_db",
        embedder: Embedder | None = None,
        answer_generator: AnswerGenerator | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.answer_generator = answer_generator or AnswerGenerator()
        self.vector_store = VectorStore(persist_directory)

    def index_pdf(self, pdf_path: str | Path) -> int:
        """Load, chunk, embed and index a PDF."""

        pages = load_pdf(pdf_path)
        chunks = chunk_pages(pages)
        embedded_records = self.embedder.embed_pdf_chunks(chunks)

        return self.vector_store.upsert_pdf_chunks(embedded_records)

    def rebuild_index(
        self,
        pdf_directory: str | Path,
        okf_directory: str | Path,
    ) -> dict[str, int]:
        """Clear both collections and rebuild them from complete directories."""
        pdf_root = Path(pdf_directory)
        if not pdf_root.is_dir():
            raise NotADirectoryError(f"PDF input directory not found: {pdf_root}")

        pdf_files = sorted(pdf_root.rglob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in {pdf_root}")

        # Read every PDF before clearing the existing collections, so corrupt
        # or unreadable input cannot destroy the currently indexed corpus.
        pdf_pages = [(path, load_pdf(path)) for path in pdf_files]
        for path, pages in pdf_pages:
             if not pages:
                raise ValueError(f"No readable pages found in PDF: {path}")
        # Validate the OKF directory before clearing the existing index.
        concepts = load_okf_directory(okf_directory)
        if not concepts:
            raise ValueError(f"No OKF concepts found in {okf_directory}")

        self.vector_store.clear()
        pdf_count = 0
        for path, pages in pdf_pages:
            chunks = chunk_pages(pages)
            embedded_records = self.embedder.embed_pdf_chunks(chunks)
            pdf_count += self.vector_store.upsert_pdf_chunks(embedded_records)

        okf_records = self.embedder.embed_okf_concepts(concepts)
        okf_count = self.vector_store.upsert_okf_concepts(okf_records)
        return {
            "pdf_files": len(pdf_files),
            "pdf_chunks": pdf_count,
            "okf_concepts": okf_count,
        }

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

    def answer_question(
        self,
        question: str,
        pdf_top_k: int = 5,
        okf_top_k: int = 5,
        final_top_k: int = 5,
    ) -> GeneratedAnswer:
        """Retrieve hybrid evidence and generate a grounded answer."""

        evidence = self.retrieve_hybrid(
            query=question,
            pdf_top_k=pdf_top_k,
            okf_top_k=okf_top_k,
            final_top_k=final_top_k,
        )

        return self.answer_generator.generate(
            question=question,
            evidence=evidence,
        )

    def collection_counts(self) -> dict[str, int]:
        return self.vector_store.collection_counts()
