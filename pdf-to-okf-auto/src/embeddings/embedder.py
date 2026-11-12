"""Embed OKF concepts and questions while preserving concept provenance."""

import json
from dataclasses import asdict, dataclass
from typing import Any

from src.ingestion.okf_loader import OKFConceptRecord

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class EmbeddedConcept:
    record_id: str
    text: str
    metadata: dict[str, Any]
    embedding: list[float]


class Embedder:
    def __init__(self, model_name=DEFAULT_EMBEDDING_MODEL, *, backend=None):
        self.model_name = model_name
        if backend is None:
            from fastembed import TextEmbedding

            backend = TextEmbedding(model_name=model_name)
        self.backend = backend

    def embed_okf_concepts(self, concepts: list[OKFConceptRecord]):
        if not concepts:
            return []
        texts = [
            "\n".join(part for part in (
                concept.title,
                concept.description,
                "Tags: " + ", ".join(concept.tags) if concept.tags else "",
                concept.content,
            ) if part)
            for concept in concepts
        ]
        vectors = list(self.backend.passage_embed(texts))
        if len(vectors) != len(concepts):
            raise RuntimeError("Embedding backend returned an unexpected vector count")
        return [
            EmbeddedConcept(
                record_id=concept.concept_id,
                text=text,
                metadata={
                    "source_type": "okf",
                    "concept_id": concept.concept_id,
                    "title": concept.title,
                    "description": concept.description,
                    "tags": json.dumps(concept.tags),
                    "source_files": json.dumps(concept.source_files),
                    "source_pages": json.dumps(concept.source_pages),
                    "evidence": json.dumps([asdict(item) for item in concept.evidence]),
                    "related_concepts": json.dumps([
                        asdict(item) for item in concept.related_concepts
                    ]),
                },
                embedding=[float(value) for value in vector],
            )
            for concept, text, vector in zip(concepts, texts, vectors)
        ]

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("Question cannot be empty")
        vectors = list(self.backend.query_embed([query]))
        if len(vectors) != 1:
            raise RuntimeError("Embedding backend returned an unexpected vector count")
        return [float(value) for value in vectors[0]]
