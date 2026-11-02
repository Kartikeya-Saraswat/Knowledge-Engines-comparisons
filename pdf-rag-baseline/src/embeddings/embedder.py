"""Generate chunk and query embeddings without coupling to vector storage."""

from dataclasses import dataclass
import os
from typing import Iterable, Protocol
import json
from src.chunking.text_chunker import ChunkRecord


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class _EmbeddingBackend(Protocol):
    def passage_embed(self, documents: list[str]) -> Iterable[object]: ...
    def query_embed(self, query: str) -> Iterable[object]: ...


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk with its original identifier, text, metadata, and vector."""

    chunk_id: str
    text: str
    metadata: dict[str, str]
    embedding: list[float]


class Embedder:
    """Embed documents and queries with one shared FastEmbed model instance."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        backend: _EmbeddingBackend | None = None,
    ) -> None:
        self.model_name = (
            model_name or os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        )
        if backend is None:
            from fastembed import TextEmbedding

            backend = TextEmbedding(model_name=self.model_name)
        self._backend = backend

    def embed_chunks(self, chunks: Iterable[ChunkRecord]) -> list[EmbeddedChunk]:
        """Embed chunk text and retain each chunk's ID and source metadata."""
        chunk_list = list(chunks)
        if not chunk_list:
            return []

        vectors = list(
        self._backend.passage_embed([chunk.text for chunk in chunk_list])
        )
        if len(vectors) != len(chunk_list):
            raise RuntimeError("Embedding backend returned an unexpected vector count")

        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata={
                    "document_name": chunk.document_name,
                    "page_numbers": json.dumps(chunk.page_numbers),
                },
                embedding=self._vector_to_list(vector),
            )
            for chunk, vector in zip(chunk_list, vectors)
        ]

    def embed_query(self, query: str) -> list[float]:
        """Embed a user query with the same model used for document chunks."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        vectors = list(self._backend.query_embed(query)) 
        if len(vectors) != 1:
            raise RuntimeError("Embedding backend returned an unexpected vector count")
        return self._vector_to_list(vectors[0])

    @staticmethod
    def _vector_to_list(vector: object) -> list[float]:
        values = vector.tolist() if hasattr(vector, "tolist") else vector
        return [float(value) for value in values]  # type: ignore[union-attr]
