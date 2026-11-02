import json

import pytest

from src.chunking.text_chunker import chunk_pages
from src.embeddings.embedder import EmbeddedChunk
from src.ingestion.pdf_loader import PageRecord
from src.pipeline import PDFRAGPipeline
from src.retrieval.vector_store import RetrievedChunk


class FakeEmbedder:
    def __init__(self):
        self.chunk_calls = []
        self.query_calls = []

    def embed_chunks(self, chunks):
        chunks = list(chunks)
        self.chunk_calls.append(chunks)
        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata={
                    "document_name": chunk.document_name,
                    "page_numbers": json.dumps(chunk.page_numbers),
                },
                embedding=[0.1, 0.2],
            )
            for chunk in chunks
        ]

    def embed_query(self, question):
        self.query_calls.append(question)
        return [0.3, 0.4]


class FakeVectorStore:
    def __init__(self, retrieved_chunks):
        self.retrieved_chunks = retrieved_chunks
        self.upsert_calls = []
        self.search_calls = []

    def upsert_chunks(self, chunks):
        self.upsert_calls.append(chunks)
        return len(chunks)

    def search(self, query_embedding, top_k, document_name=None):
        self.search_calls.append((query_embedding, top_k, document_name))
        return self.retrieved_chunks


class FakeAnswerGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, question, chunks):
        self.calls.append((question, list(chunks)))
        return "The PDF says hello [sample.pdf, p. 1]."


def test_run_connects_all_pipeline_stages(monkeypatch):
    pages = [PageRecord("sample.pdf", 1, "hello from the sample pdf")]
    retrieved = [
        RetrievedChunk(
            chunk_id="retrieved-1",
            text="hello from the sample pdf",
            document_name="sample.pdf",
            page_numbers=(1,),
            distance=0.05,
        )
    ]
    monkeypatch.setattr("src.pipeline.load_pdf", lambda _path: pages)
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore(retrieved)
    answer_generator = FakeAnswerGenerator()
    pipeline = PDFRAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        answer_generator=answer_generator,
        chunk_size=20,
        overlap=2,
        top_k=3,
    )

    result = pipeline.run("sample.pdf", "What does the PDF say?")

    assert result.answer == "The PDF says hello [sample.pdf, p. 1]."
    assert result.indexed_chunks == 1
    assert result.retrieved_chunks == tuple(retrieved)
    assert embedder.chunk_calls == [chunk_pages(pages, chunk_size=20, overlap=2)]
    assert vector_store.upsert_calls[0][0].text == "hello from the sample pdf"
    assert embedder.query_calls == ["What does the PDF say?"]
    assert vector_store.search_calls == [([0.3, 0.4], 3, None)]
    assert answer_generator.calls == [("What does the PDF say?", retrieved)]


def test_index_pdf_returns_zero_when_no_text_pages(monkeypatch):
    monkeypatch.setattr("src.pipeline.load_pdf", lambda _path: [])
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore([])
    pipeline = PDFRAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        answer_generator=FakeAnswerGenerator(),
    )

    assert pipeline.index_pdf("scanned.pdf") == 0
    assert embedder.chunk_calls == []
    assert vector_store.upsert_calls == []


def test_answer_question_accepts_document_filter():
    retrieved = [
        RetrievedChunk("id", "text", "sample.pdf", (2,), 0.1),
    ]
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore(retrieved)
    pipeline = PDFRAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        answer_generator=FakeAnswerGenerator(),
        top_k=4,
    )

    result = pipeline.answer_question("Question?", document_name="sample.pdf")

    assert result.answer.startswith("The PDF says hello")
    assert vector_store.search_calls == [([0.3, 0.4], 4, "sample.pdf")]


def test_blank_question_is_rejected_before_indexing(monkeypatch):
    called = []
    monkeypatch.setattr("src.pipeline.load_pdf", lambda _path: called.append(_path))
    pipeline = PDFRAGPipeline(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore([]),
        answer_generator=FakeAnswerGenerator(),
    )

    with pytest.raises(ValueError, match="question cannot be empty"):
        pipeline.run("sample.pdf", "  ")

    assert called == []

def test_answer_question_accepts_top_k_override():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore([])
    pipeline = PDFRAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        answer_generator=FakeAnswerGenerator(),
        top_k=5,
    )

    pipeline.answer_question("Question?", top_k=2)

    assert vector_store.search_calls == [([0.3, 0.4], 2, None)]


def test_answer_question_rejects_invalid_top_k():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore([])
    pipeline = PDFRAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        answer_generator=FakeAnswerGenerator(),
    )

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        pipeline.answer_question("Question?", top_k=0)

    assert embedder.query_calls == []