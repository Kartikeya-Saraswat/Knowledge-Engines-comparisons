import json

import pytest

from src.embeddings import embedder as embedder_module
from src.embeddings.embedder import Embedder
from src.chunking.text_chunker import ChunkRecord
from src.ingestion.okf_loader import (
    OKFConceptRecord,
    EvidenceRecord,
    RelatedConcept,
)


class FakeTextEmbedding:
    def __init__(self, model_name):
        self.model_name = model_name

    def passage_embed(self, texts):
        for _ in texts:
            yield [0.1, 0.2, 0.3]

    def query_embed(self, texts):
        for _ in texts:
            yield [0.4, 0.5, 0.6]


@pytest.fixture
def test_embedder(monkeypatch):
    monkeypatch.setattr(
        embedder_module,
        "TextEmbedding",
        FakeTextEmbedding,
    )
    return Embedder(model_name="test-model")


def test_embed_pdf_chunks(test_embedder):
    chunks = [
        ChunkRecord(
            chunk_id="chunk-001",
            document_name="sample.pdf",
            text="Sample PDF content.",
            page_numbers=(1, 2),
        )
    ]

    records = test_embedder.embed_pdf_chunks(chunks)

    assert len(records) == 1
    assert records[0].record_id == "chunk-001"
    assert records[0].text == "Sample PDF content."
    assert records[0].metadata["source_type"] == "pdf"
    assert records[0].metadata["document_name"] == "sample.pdf"
    assert json.loads(records[0].metadata["page_numbers"]) == [1, 2]
    assert records[0].embedding == (0.1, 0.2, 0.3)


def test_embed_okf_concepts(test_embedder):
    concepts = [
        OKFConceptRecord(
            concept_id="KC-001",
            title="Resource Planning",
            description="Planning resources in India.",
            tags=("economics", "planning"),
            content="Resource planning is important.",
            source_files=("sample.pdf",),
            source_pages=(3,),
            evidence=(
                EvidenceRecord(
                    page_number=3,
                    quote="Resource planning is important.",
                ),
            ),
            related_concepts=(
                RelatedConcept(
                    title="Economic Development",
                    filename="economic-development.md",
                ),
            ),
        )
    ]

    records = test_embedder.embed_okf_concepts(concepts)

    assert len(records) == 1

    record = records[0]

    assert record.record_id == "KC-001"
    assert record.metadata["source_type"] == "okf"
    assert record.metadata["concept_id"] == "KC-001"
    assert "Resource Planning" in record.text

    assert json.loads(record.metadata["tags"]) == [
        "economics",
        "planning",
    ]
    assert json.loads(record.metadata["source_pages"]) == [3]
    assert json.loads(record.metadata["evidence"])[0]["page_number"] == 3
    assert json.loads(record.metadata["related_concepts"])[0]["title"] == (
        "Economic Development"
    )


def test_embed_query(test_embedder):
    vector = test_embedder.embed_query("What is resource planning?")

    assert vector == (0.4, 0.5, 0.6)


def test_embed_multiple_queries(test_embedder):
    vectors = test_embedder.embed_queries([
        "What is resource planning?",
        "Explain economic development.",
    ])

    assert len(vectors) == 2
    assert vectors[0] == (0.4, 0.5, 0.6)


def test_empty_passages_return_empty_list(test_embedder):
    assert test_embedder.embed_pdf_chunks([]) == []
    assert test_embedder.embed_okf_concepts([]) == []


def test_empty_query_is_rejected(test_embedder):
    with pytest.raises(ValueError):
        test_embedder.embed_query("   ")


def test_empty_query_in_batch_is_rejected(test_embedder):
    with pytest.raises(ValueError):
        test_embedder.embed_queries(["valid query", ""])
        