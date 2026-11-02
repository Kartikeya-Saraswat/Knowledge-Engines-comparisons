import pytest

from src.chunking.text_chunker import ChunkRecord
from src.embeddings.embedder import DEFAULT_EMBEDDING_MODEL, Embedder


class FakeBackend:
    def __init__(self):
        self.passage_calls = []
        self.query_calls = []

    def passage_embed(self, documents):
        self.passage_calls.append(documents)
        return [[float(index), 1.0] for index, _ in enumerate(documents)]

    def query_embed(self, query):
        self.query_calls.append(query)
        return [[0.0, 1.0]]


def test_default_model_is_exact_supported_identifier():
    backend = FakeBackend()

    embedder = Embedder(backend=backend)

    assert embedder.model_name == "BAAI/bge-small-en-v1.5"
    assert embedder.model_name == DEFAULT_EMBEDDING_MODEL


def test_embed_chunks_preserves_ids_text_and_metadata():
    backend = FakeBackend()
    embedder = Embedder(backend=backend)
    chunks = [
        ChunkRecord("chunk-1", "sample.pdf", "first chunk", (1, 2)),
        ChunkRecord("chunk-2", "sample.pdf", "second chunk", (2,)),
    ]

    embedded = embedder.embed_chunks(chunks)

    assert backend.passage_calls == [["first chunk", "second chunk"]]
    assert backend.query_calls == []
    assert [item.chunk_id for item in embedded] == ["chunk-1", "chunk-2"]
    assert [item.text for item in embedded] == ["first chunk", "second chunk"]
    assert embedded[0].metadata == {
        "document_name": "sample.pdf",
        "page_numbers": "[1, 2]",
    }   
    assert embedded[0].embedding == [0.0, 1.0]


def test_query_uses_same_backend_as_chunk_embedding():
    backend = FakeBackend()
    embedder = Embedder(backend=backend)

    vector = embedder.embed_query("find the first chunk")

    assert backend.query_calls == ["find the first chunk"]
    assert backend.passage_calls == []
    assert vector == [0.0, 1.0]


def test_empty_chunk_list_does_not_call_backend():
    backend = FakeBackend()

    assert Embedder(backend=backend).embed_chunks([]) == []
    assert backend.passage_calls == []
    assert backend.query_calls == []


def test_empty_query_is_rejected():
    with pytest.raises(ValueError, match="query cannot be empty"):
        Embedder(backend=FakeBackend()).embed_query("  ")

def test_unexpected_embedding_count_raises():
    class InvalidBackend:
        def passage_embed(self, documents):
            return []

        def query_embed(self, query):
            return []

    embedder = Embedder(backend=InvalidBackend())

    with pytest.raises(RuntimeError):
        embedder.embed_query("test query")

def test_unexpected_chunk_embedding_count_raises():
    class InvalidBackend:
        def passage_embed(self, documents):
            return []

        def query_embed(self, query):
            return []

    chunks = [ChunkRecord("chunk-1", "sample.pdf", "hello", (1,))]
    embedder = Embedder(backend=InvalidBackend())

    with pytest.raises(RuntimeError):
        embedder.embed_chunks(chunks)