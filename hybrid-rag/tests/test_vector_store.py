import pytest

from src.embeddings.embedder import EmbeddedRecord
from src.retrieval.vector_store import (
    VectorStore,
    PDF_COLLECTION,
    OKF_COLLECTION,
)


@pytest.fixture
def vector_store(tmp_path):
    return VectorStore(persist_directory=tmp_path / "chroma_test")


def make_record(
    record_id,
    source_type,
    text,
    embedding,
    metadata=None,
):
    record_metadata = {
        "source_type": source_type,
    }

    if metadata:
        record_metadata.update(metadata)

    return EmbeddedRecord(
        record_id=record_id,
        text=text,
        metadata=record_metadata,
        embedding=tuple(embedding),
    )


def test_collections_are_created(vector_store):
    counts = vector_store.collection_counts()

    assert counts[PDF_COLLECTION] == 0
    assert counts[OKF_COLLECTION] == 0


def test_upsert_pdf_chunks(vector_store):
    records = [
        make_record(
            "pdf-001",
            "pdf",
            "PDF content about resource planning.",
            [1.0, 0.0, 0.0],
            {"document_name": "sample.pdf"},
        )
    ]

    inserted = vector_store.upsert_pdf_chunks(records)

    assert inserted == 1
    assert vector_store.collection_counts()[PDF_COLLECTION] == 1


def test_upsert_okf_concepts(vector_store):
    records = [
        make_record(
            "KC-001",
            "okf",
            "OKF concept about resource planning.",
            [1.0, 0.0, 0.0],
            {"concept_id": "KC-001"},
        )
    ]

    inserted = vector_store.upsert_okf_concepts(records)

    assert inserted == 1
    assert vector_store.collection_counts()[OKF_COLLECTION] == 1


def test_pdf_and_okf_collections_are_independent(vector_store):
    pdf_record = make_record(
        "record-001",
        "pdf",
        "PDF content.",
        [1.0, 0.0, 0.0],
    )

    okf_record = make_record(
        "record-001",
        "okf",
        "OKF content.",
        [1.0, 0.0, 0.0],
    )

    vector_store.upsert_pdf_chunks([pdf_record])
    vector_store.upsert_okf_concepts([okf_record])

    counts = vector_store.collection_counts()

    assert counts[PDF_COLLECTION] == 1
    assert counts[OKF_COLLECTION] == 1


def test_search_pdf_returns_relevant_records(vector_store):
    records = [
        make_record(
            "pdf-001",
            "pdf",
            "Resource planning in India.",
            [1.0, 0.0, 0.0],
            {"document_name": "sample.pdf"},
        ),
        make_record(
            "pdf-002",
            "pdf",
            "Unrelated content.",
            [0.0, 1.0, 0.0],
        ),
    ]

    vector_store.upsert_pdf_chunks(records)

    results = vector_store.search_pdf(
        query_embedding=(1.0, 0.0, 0.0),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].record_id == "pdf-001"
    assert results[0].metadata["document_name"] == "sample.pdf"
    assert results[0].distance <= results[1].distance


def test_search_okf_returns_concept_metadata(vector_store):
    records = [
        make_record(
            "KC-001",
            "okf",
            "Resource planning concept.",
            [1.0, 0.0, 0.0],
            {
                "concept_id": "KC-001",
                "title": "Resource Planning",
            },
        )
    ]

    vector_store.upsert_okf_concepts(records)

    results = vector_store.search_okf(
        query_embedding=(1.0, 0.0, 0.0),
    )

    assert len(results) == 1
    assert results[0].record_id == "KC-001"
    assert results[0].metadata["title"] == "Resource Planning"


def test_repeated_upsert_does_not_create_duplicates(vector_store):
    record = make_record(
        "pdf-001",
        "pdf",
        "Original content.",
        [1.0, 0.0, 0.0],
    )

    vector_store.upsert_pdf_chunks([record])

    updated_record = make_record(
        "pdf-001",
        "pdf",
        "Updated content.",
        [1.0, 0.0, 0.0],
    )

    vector_store.upsert_pdf_chunks([updated_record])

    results = vector_store.search_pdf(
        query_embedding=(1.0, 0.0, 0.0)
    )

    assert vector_store.collection_counts()[PDF_COLLECTION] == 1
    assert results[0].text == "Updated content."


def test_empty_upsert(vector_store):
    assert vector_store.upsert_pdf_chunks([]) == 0
    assert vector_store.upsert_okf_concepts([]) == 0


def test_empty_collection_search(vector_store):
    results = vector_store.search_pdf(
        query_embedding=(1.0, 0.0, 0.0)
    )

    assert results == []


def test_invalid_source_type_is_rejected(vector_store):
    record = make_record(
        "KC-001",
        "okf",
        "OKF content.",
        [1.0, 0.0, 0.0],
    )

    with pytest.raises(ValueError):
        vector_store.upsert_pdf_chunks([record])


def test_invalid_top_k_is_rejected(vector_store):
    with pytest.raises(ValueError):
        vector_store.search_pdf(
            query_embedding=(1.0, 0.0, 0.0),
            top_k=0,
        )