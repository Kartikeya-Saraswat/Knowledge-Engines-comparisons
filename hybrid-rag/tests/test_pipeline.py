import pytest

from src.pipeline import HybridRAGPipeline
from src.embeddings.embedder import EmbeddedRecord
from src.ingestion.pdf_loader import PageRecord
from src.ingestion.okf_loader import (
    OKFConceptRecord,
    EvidenceRecord,
)


class FakeEmbedder:

    def embed_pdf_chunks(self, chunks):
        return [
            EmbeddedRecord(
                record_id=chunk.chunk_id,
                text=chunk.text,
                metadata={
                    "source_type": "pdf",
                    "document_name": chunk.document_name,
                    "page_numbers": str(chunk.page_numbers),
                },
                embedding=(1.0, 0.0, 0.0),
            )
            for chunk in chunks
        ]

    def embed_okf_concepts(self, concepts):
        return [
            EmbeddedRecord(
                record_id=concept.concept_id,
                text=concept.title + "\n" + concept.content,
                metadata={
                    "source_type": "okf",
                    "concept_id": concept.concept_id,
                    "title": concept.title,
                },
                embedding=(1.0, 0.0, 0.0),
            )
            for concept in concepts
        ]

    def embed_query(self, query):
        return (1.0, 0.0, 0.0)


@pytest.fixture
def pipeline(tmp_path):
    return HybridRAGPipeline(
        persist_directory=tmp_path / "chroma_test",
        embedder=FakeEmbedder(),
    )


def test_index_and_retrieve_pdf(pipeline, monkeypatch):
    import src.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "load_pdf",
        lambda path: [
            PageRecord(
                document_name="sample.pdf",
                page_number=1,
                text="Resource planning in India.",
            )
        ],
    )

    indexed = pipeline.index_pdf("sample.pdf")

    results = pipeline.retrieve_pdf(
        "What is resource planning?",
        top_k=3,
    )

    assert indexed == 1
    assert len(results) == 1
    assert results[0].metadata["document_name"] == "sample.pdf"


def test_index_and_retrieve_okf(pipeline, monkeypatch):
    import src.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "load_okf_directory",
        lambda directory: [
            OKFConceptRecord(
                concept_id="KC-001",
                title="Resource Planning",
                description="Planning resources.",
                tags=("economics",),
                content="Resource planning in India.",
                source_files=("sample.pdf",),
                source_pages=(3,),
                evidence=(
                    EvidenceRecord(
                        page_number=3,
                        quote="Resource planning in India.",
                    ),
                ),
                related_concepts=(),
            )
        ],
    )

    indexed = pipeline.index_okf_directory("sample_okf")

    results = pipeline.retrieve_okf(
        "What is resource planning?",
        top_k=3,
    )

    assert indexed == 1
    assert len(results) == 1
    assert results[0].metadata["concept_id"] == "KC-001"


def test_pdf_and_okf_remain_independent(pipeline, monkeypatch):
    import src.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "load_pdf",
        lambda path: [
            PageRecord(
                "sample.pdf",
                1,
                "PDF resource planning content.",
            )
        ],
    )

    monkeypatch.setattr(
        pipeline_module,
        "load_okf_directory",
        lambda directory: [
            OKFConceptRecord(
                concept_id="KC-001",
                title="Resource Planning",
                description="Planning resources.",
                tags=(),
                content="OKF resource planning content.",
                source_files=("sample.pdf",),
                source_pages=(3,),
                evidence=(),
                related_concepts=(),
            )
        ],
    )

    pipeline.index_pdf("sample.pdf")
    pipeline.index_okf_directory("sample_okf")

    counts = pipeline.collection_counts()

    assert counts["hybrid_pdf_chunks"] == 1
    assert counts["hybrid_okf_concepts"] == 1


def test_repeated_indexing_does_not_duplicate_records(
    pipeline,
    monkeypatch,
):
    import src.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "load_pdf",
        lambda path: [
            PageRecord(
                "sample.pdf",
                1,
                "Resource planning in India.",
            )
        ],
    )

    pipeline.index_pdf("sample.pdf")
    pipeline.index_pdf("sample.pdf")

    assert pipeline.collection_counts()["hybrid_pdf_chunks"] == 1


def test_retrieval_returns_empty_for_unindexed_source(pipeline):
    assert pipeline.retrieve_pdf("resource planning") == []
    assert pipeline.retrieve_okf("resource planning") == []

def test_hybrid_retrieval_combines_pdf_and_okf(
    pipeline,
    monkeypatch,
):
    import src.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "load_pdf",
        lambda path: [
            PageRecord(
                "sample.pdf",
                1,
                "PDF resource planning content.",
            )
        ],
    )

    monkeypatch.setattr(
        pipeline_module,
        "load_okf_directory",
        lambda directory: [
            OKFConceptRecord(
                concept_id="KC-001",
                title="Resource Planning",
                description="Planning resources.",
                tags=(),
                content="OKF resource planning content.",
                source_files=("sample.pdf",),
                source_pages=(3,),
                evidence=(),
                related_concepts=(),
            )
        ],
    )

    pipeline.index_pdf("sample.pdf")
    pipeline.index_okf_directory("sample_okf")

    results = pipeline.retrieve_hybrid(
        "What is resource planning?",
        pdf_top_k=5,
        okf_top_k=5,
        final_top_k=5,
    )

    assert len(results) == 2

    source_types = {
        result.metadata["source_type"]
        for result in results
    }

    assert source_types == {"pdf", "okf"}


def test_hybrid_retrieval_respects_final_top_k(pipeline, monkeypatch):
    import src.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "load_pdf",
        lambda path: [
            PageRecord(
                "sample.pdf",
                1,
                "PDF resource planning content.",
            )
        ],
    )

    monkeypatch.setattr(
        pipeline_module,
        "load_okf_directory",
        lambda directory: [
            OKFConceptRecord(
                concept_id="KC-001",
                title="Resource Planning",
                description="Planning resources.",
                tags=(),
                content="OKF resource planning content.",
                source_files=("sample.pdf",),
                source_pages=(3,),
                evidence=(),
                related_concepts=(),
            )
        ],
    )

    pipeline.index_pdf("sample.pdf")
    pipeline.index_okf_directory("sample_okf")

    results = pipeline.retrieve_hybrid(
        "What is resource planning?",
        final_top_k=1,
    )

    assert len(results) == 1


def test_hybrid_retrieval_rejects_invalid_final_top_k(pipeline):
    with pytest.raises(ValueError):
        pipeline.retrieve_hybrid(
            "resource planning",
            final_top_k=0,
        )