"""A dedicated persistent vector collection containing only OKF concepts."""

from dataclasses import dataclass
from typing import Any

from src.embeddings.embedder import DEFAULT_EMBEDDING_MODEL, EmbeddedConcept

COLLECTION_NAME = "okf_concepts"


@dataclass(frozen=True)
class RetrievedConcept:
    record_id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class VectorStore:
    def __init__(self, persist_directory, *, model_name=DEFAULT_EMBEDDING_MODEL,
                 client=None):
        if client is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "embedding_model": model_name},
            embedding_function=None,
        )
        metadata = self.collection.metadata or {}
        metric = metadata.get("hnsw:space")
        if metric is None:
            metric = self.collection.configuration.get("hnsw", {}).get("space")
        if metric != "cosine":
            raise ValueError("OKF index must use cosine distance")
        if metadata.get("embedding_model") != model_name:
            raise ValueError("Embedding model differs from the stored OKF index")

    def replace_concepts(self, records: list[EmbeddedConcept]) -> int:
        """Update a complete corpus and remove concepts absent from the new corpus."""
        if not records:
            raise ValueError("Cannot index an empty OKF corpus")
        ids = [record.record_id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate OKF concept IDs")
        if any(record.metadata.get("source_type") != "okf" for record in records):
            raise ValueError("The OKF index accepts only OKF concepts")
        previous_ids = set(self.collection.get(include=[])["ids"])
        self.collection.upsert(
            ids=ids,
            documents=[record.text for record in records],
            metadatas=[record.metadata for record in records],
            embeddings=[record.embedding for record in records],
        )
        stale = previous_ids - set(ids)
        if stale:
            self.collection.delete(ids=sorted(stale))
        return len(records)

    def search(self, query_embedding, top_k=5) -> list[RetrievedConcept]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedConcept(record_id, text or "", metadata or {}, float(distance))
            for record_id, text, metadata, distance in zip(
                result["ids"][0], result["documents"][0],
                result["metadatas"][0], result["distances"][0],
            )
        ]
