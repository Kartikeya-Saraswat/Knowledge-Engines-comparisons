"""End-to-end PDF retrieval-augmented generation pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path

from src.chunking.text_chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_pages,
)
from src.embeddings.embedder import Embedder
from src.generation.answer_generator import AnswerGenerator
from src.ingestion.pdf_loader import load_pdf
from src.retrieval.vector_store import RetrievedChunk, VectorStore


@dataclass(frozen=True)
class PipelineResult:
    """Answer and retrieval details produced by a pipeline query."""

    answer: str
    retrieved_chunks: tuple[RetrievedChunk, ...]
    indexed_chunks: int = 0


class PDFRAGPipeline:
    """Connect PDF extraction, chunking, embedding, retrieval, and generation."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        answer_generator: AnswerGenerator | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        top_k: int | None = None,
    ) -> None:
        if embedder is None or vector_store is None or answer_generator is None:
            from dotenv import load_dotenv

            load_dotenv()

        self.chunk_size = (
            chunk_size
            if chunk_size is not None
            else int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
        )
        self.overlap = (
            overlap
            if overlap is not None
            else int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))
        )
        self.top_k = top_k if top_k is not None else int(os.getenv("TOP_K", 5))
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        self.embedder = embedder if embedder is not None else Embedder()
        self.vector_store = (
            vector_store if vector_store is not None else VectorStore()
        )
        self.answer_generator = (
            answer_generator if answer_generator is not None else AnswerGenerator()
        )

    def index_pdf(self, pdf_path: str | Path) -> int:
        """Extract, chunk, embed, and store one PDF; return chunks indexed."""
        pages = load_pdf(pdf_path)
        chunks = chunk_pages(pages, chunk_size=self.chunk_size, overlap=self.overlap)
        if not chunks:
            return 0
        embedded_chunks = self.embedder.embed_chunks(chunks)
        return self.vector_store.upsert_chunks(embedded_chunks)

    def answer_question(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_name: str | None = None,
    ) -> PipelineResult:
        """Retrieve relevant chunks and generate a grounded answer."""
        if not question.strip():
            raise ValueError("question cannot be empty")

        effective_top_k = self.top_k if top_k is None else top_k

        if effective_top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self.embedder.embed_query(question)

        retrieved_chunks = self.vector_store.search(
            query_embedding,
            top_k=effective_top_k,
            document_name=document_name,
        )
        answer = self.answer_generator.generate(question, retrieved_chunks)
        return PipelineResult(
            answer=answer,
            retrieved_chunks=tuple(retrieved_chunks),
        )

    def run(
        self,
        pdf_path: str | Path,
        question: str,
        *,
        top_k: int | None = None,
        document_name: str | None = None,
    ) -> PipelineResult:
        """Index a PDF and answer one question using the stored corpus."""
        if not question.strip():
            raise ValueError("question cannot be empty")

        indexed_count = self.index_pdf(pdf_path)
        result = self.answer_question(
            question,
            top_k=top_k,
            document_name=document_name,
        )
        return PipelineResult(
            answer=result.answer,
            retrieved_chunks=result.retrieved_chunks,
            indexed_chunks=indexed_count,
        )
