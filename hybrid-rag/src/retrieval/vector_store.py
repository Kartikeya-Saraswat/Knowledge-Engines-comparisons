import chromadb

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.embeddings.embedder import EmbeddedRecord


PDF_COLLECTION = "hybrid_pdf_chunks"
OKF_COLLECTION = "hybrid_okf_concepts"


@dataclass(frozen=True)
class RetrievedRecord:
    record_id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class VectorStore:
    def __init__(
        self,
        persist_directory: str | Path = "data/chroma_db",
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.pdf_collection = self.client.get_or_create_collection(
            name=PDF_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        self.okf_collection = self.client.get_or_create_collection(
            name=OKF_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def _upsert(
        self,
        collection,
        records: list[EmbeddedRecord],
        expected_source_type: str,
    ) -> int:
        """Insert or update embedded records in a collection."""

        if not records:
            return 0

        for record in records:
            if record.metadata.get("source_type") != expected_source_type:
                raise ValueError(
                    f"Expected source_type='{expected_source_type}', "
                    f"got '{record.metadata.get('source_type')}'"
                )

        collection.upsert(
            ids=[record.record_id for record in records],
            documents=[record.text for record in records],
            metadatas=[record.metadata for record in records],
            embeddings=[
                list(record.embedding) for record in records
            ],
        )

        return len(records)

    def upsert_pdf_chunks(
        self,
        records: list[EmbeddedRecord],
    ) -> int:
        return self._upsert(
            self.pdf_collection,
            records,
            expected_source_type="pdf",
        )

    def upsert_okf_concepts(
        self,
        records: list[EmbeddedRecord],
    ) -> int:
        return self._upsert(
            self.okf_collection,
            records,
            expected_source_type="okf",
        )

    def _search(
        self,
        collection,
        query_embedding: tuple[float, ...],
        top_k: int,
    ) -> list[RetrievedRecord]:
        """Search a collection using a query embedding."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")

        if collection.count() == 0:
            return []

        results = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for record_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            retrieved.append(
                RetrievedRecord(
                    record_id=record_id,
                    text=document,
                    metadata=metadata,
                    distance=float(distance),
                )
            )

        return retrieved

    def search_pdf(
        self,
        query_embedding: tuple[float, ...],
        top_k: int = 5,
    ) -> list[RetrievedRecord]:
        return self._search(
            self.pdf_collection,
            query_embedding,
            top_k,
        )

    def search_okf(
        self,
        query_embedding: tuple[float, ...],
        top_k: int = 5,
    ) -> list[RetrievedRecord]:
        return self._search(
            self.okf_collection,
            query_embedding,
            top_k,
        )

    def collection_counts(self) -> dict[str, int]:
        """Return the number of records in each collection."""

        return {
            PDF_COLLECTION: self.pdf_collection.count(),
            OKF_COLLECTION: self.okf_collection.count(),
        }