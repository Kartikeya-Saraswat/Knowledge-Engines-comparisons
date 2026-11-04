import json
from dataclasses import dataclass
from typing import Any

from fastembed import TextEmbedding

from src.chunking.text_chunker import ChunkRecord
from src.ingestion.okf_loader import OKFConceptRecord


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class EmbeddedRecord:
    record_id: str
    text: str
    metadata: dict[str, Any]
    embedding: tuple[float, ...]


class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.model = TextEmbedding(model_name=model_name)

    def _embed_passages(self, texts: list[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []

        vectors = self.model.passage_embed(texts)
        return [tuple(float(value) for value in vector) for vector in vectors]

    def embed_pdf_chunks(
        self,
        chunks: list[ChunkRecord],
    ) -> list[EmbeddedRecord]:
        """Embed PDF chunks while retaining their source metadata."""

        texts = [chunk.text for chunk in chunks]
        vectors = self._embed_passages(texts)

        records = []

        for chunk, vector in zip(chunks, vectors):
            records.append(
                EmbeddedRecord(
                    record_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata={
                        "source_type": "pdf",
                        "document_name": chunk.document_name,
                        "page_numbers": json.dumps(chunk.page_numbers),
                    },
                    embedding=vector,
                )
            )

        return records

    def embed_okf_concepts(
        self,
        concepts: list[OKFConceptRecord],
    ) -> list[EmbeddedRecord]:
        """Embed OKF concepts while retaining their knowledge metadata."""

        texts = [
            "\n".join(
                part for part in [
                    concept.title,
                    concept.description,
                    "Tags: " + ", ".join(concept.tags) if concept.tags else "",
                    concept.content,
                ]
                if part
            )
            for concept in concepts
        ]

        vectors = self._embed_passages(texts)

        records = []

        for concept, embedding_text, vector in zip(
            concepts, texts, vectors
        ):
            records.append(
                EmbeddedRecord(
                    record_id=concept.concept_id,
                    text=embedding_text,
                    metadata={
                        "source_type": "okf",
                        "concept_id": concept.concept_id,
                        "title": concept.title,
                        "description": concept.description,
                        "tags": json.dumps(concept.tags),
                        "source_files": json.dumps(concept.source_files),
                        "source_pages": json.dumps(concept.source_pages),
                        "evidence": json.dumps([
                            {
                                "page_number": item.page_number,
                                "quote": item.quote,
                            }
                            for item in concept.evidence
                        ]),
                        "related_concepts": json.dumps([
                            {
                                "title": item.title,
                                "filename": item.filename,
                            }
                            for item in concept.related_concepts
                        ]),
                    },
                    embedding=vector,
                )
            )

        return records

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Embed a user query using the model's query-specific method."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        vectors = self.model.query_embed([query])
        return tuple(float(value) for value in next(iter(vectors)))

    def embed_queries(self, queries: list[str]) -> list[tuple[float, ...]]:
        """Embed multiple user queries."""

        if not queries:
            return []

        if any(not query.strip() for query in queries):
            raise ValueError("Queries cannot contain empty strings")

        vectors = self.model.query_embed(queries)

        return [
            tuple(float(value) for value in vector)
            for vector in vectors
        ]