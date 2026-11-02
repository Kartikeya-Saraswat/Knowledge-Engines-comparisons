"""Persistent ChromaDB storage and vector retrieval for embedded chunks."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol

from src.embeddings.embedder import EmbeddedChunk


DEFAULT_COLLECTION_NAME = "pdf_rag_chunks"
DEFAULT_PERSIST_DIRECTORY = Path(__file__).resolve().parents[2] / "chroma_db"
DISTANCE_METRIC = "cosine"


class _Collection(Protocol):
    metadata: dict[str, Any] | None

    def upsert(self, **kwargs: Any) -> Any: ...

    def count(self) -> int: ...

    def query(self, **kwargs: Any) -> dict[str, Any]: ...


class _Client(Protocol):
    def get_or_create_collection(self, **kwargs: Any) -> _Collection: ...


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved document chunk with its source provenance and distance."""

    chunk_id: str
    text: str
    document_name: str
    page_numbers: tuple[int, ...]
    distance: float


class VectorStore:
    """Store embeddings in a persistent local ChromaDB cosine collection."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        *,
        client: _Client | None = None,
    ) -> None:
        if client is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(persist_directory))

        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": DISTANCE_METRIC},
            embedding_function=None,
        )

        collection_metadata = self._collection.metadata or {}
        if collection_metadata.get("hnsw:space") != DISTANCE_METRIC:
            raise ValueError(
                f"Collection {collection_name!r} must use "
                f"hnsw:space={DISTANCE_METRIC!r}; its metric cannot be changed "
                "after creation. Use a new collection name or recreate it."
            )

    def upsert_chunks(self, chunks: list[EmbeddedChunk]) -> int:
        """Persist chunk IDs, text, vectors, and metadata; return count stored."""
        if not chunks:
            return 0

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the nearest chunks, optionally filtered by source document."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")

        count = self._collection.count()
        if count == 0:
            return []

        query: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if document_name is not None:
            query["where"] = {"document_name": document_name}

        results = self._collection.query(**query)
        ids = results["ids"][0]
        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []

        retrieved: list[RetrievedChunk] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}
            pages_value = metadata.get("page_numbers", "[]")
            page_numbers = (
                json.loads(pages_value)
                if isinstance(pages_value, str)
                else pages_value
            )
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=documents[index] or "",
                    document_name=metadata.get("document_name", ""),
                    page_numbers=tuple(int(page) for page in page_numbers),
                    distance=float(distances[index]),
                )
            )
        return retrieved
