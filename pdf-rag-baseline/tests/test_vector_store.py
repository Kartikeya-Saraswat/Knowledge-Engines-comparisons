import json
import sys
from types import SimpleNamespace

import pytest

from src.embeddings.embedder import EmbeddedChunk
from src.retrieval.vector_store import DISTANCE_METRIC, VectorStore


class FakeCollection:
    def __init__(self, metadata=None):
        self.metadata = metadata or {"hnsw:space": DISTANCE_METRIC}
        self.upsert_call = None
        self.query_call = None
        self.records = []

    def upsert(self, **kwargs):
        self.upsert_call = kwargs
        self.records = list(zip(kwargs["ids"], kwargs["documents"], kwargs["metadatas"]))

    def count(self):
        return len(self.records)

    def query(self, **kwargs):
        self.query_call = kwargs
        selected = self.records
        where = kwargs.get("where")
        if where:
            selected = [
                record for record in selected
                if record[2].get("document_name") == where["document_name"]
            ]
        selected = selected[:kwargs["n_results"]]
        return {
            "ids": [[record[0] for record in selected]],
            "documents": [[record[1] for record in selected]],
            "metadatas": [[record[2] for record in selected]],
            "distances": [[0.1 + index for index, _ in enumerate(selected)]],
        }


class FakeClient:
    def __init__(self, collection):
        self.collection = collection
        self.create_call = None

    def get_or_create_collection(self, **kwargs):
        self.create_call = kwargs
        return self.collection


def make_chunk(chunk_id, document_name, text, pages, embedding):
    return EmbeddedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "document_name": document_name,
            "page_numbers": json.dumps(pages),
        },
        embedding=embedding,
    )


def test_collection_is_configured_for_cosine():
    collection = FakeCollection()
    client = FakeClient(collection)

    VectorStore(client=client)

    assert client.create_call["metadata"] == {"hnsw:space": "cosine"}
    assert client.create_call["embedding_function"] is None


def test_default_client_uses_persistent_local_path(tmp_path, monkeypatch):
    client = FakeClient(FakeCollection())
    calls = []

    def persistent_client_factory(path):
        calls.append(path)
        return client

    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        SimpleNamespace(PersistentClient=persistent_client_factory),
    )

    VectorStore(persist_directory=tmp_path)

    assert calls == [str(tmp_path)]


def test_incompatible_existing_collection_is_rejected():
    client = FakeClient(FakeCollection({"hnsw:space": "l2"}))

    with pytest.raises(ValueError, match="must use hnsw:space='cosine'"):
        VectorStore(client=client)


def test_upsert_stores_ids_text_vectors_and_metadata():
    collection = FakeCollection()
    store = VectorStore(client=FakeClient(collection))
    chunks = [make_chunk("id-1", "one.pdf", "text one", [1, 2], [0.1, 0.2])]

    assert store.upsert_chunks(chunks) == 1
    assert collection.upsert_call == {
        "ids": ["id-1"],
        "documents": ["text one"],
        "embeddings": [[0.1, 0.2]],
        "metadatas": [{"document_name": "one.pdf", "page_numbers": "[1, 2]"}],
    }


def test_search_returns_top_k_with_provenance():
    collection = FakeCollection()
    store = VectorStore(client=FakeClient(collection))
    store.upsert_chunks(
        [
            make_chunk("id-1", "one.pdf", "first text", [1, 3], [0.1, 0.2]),
            make_chunk("id-2", "two.pdf", "second text", [4], [0.3, 0.4]),
        ]
    )

    results = store.search([0.2, 0.1], top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "id-1"
    assert results[0].text == "first text"
    assert results[0].document_name == "one.pdf"
    assert results[0].page_numbers == (1, 3)
    assert results[0].distance == 0.1
    assert collection.query_call["query_embeddings"] == [[0.2, 0.1]]


def test_search_filters_by_document_name():
    collection = FakeCollection()
    store = VectorStore(client=FakeClient(collection))
    store.upsert_chunks(
        [
            make_chunk("id-1", "one.pdf", "first text", [1], [0.1]),
            make_chunk("id-2", "two.pdf", "second text", [2], [0.2]),
        ]
    )

    results = store.search([0.3], document_name="two.pdf")

    assert [result.document_name for result in results] == ["two.pdf"]
    assert collection.query_call["where"] == {"document_name": "two.pdf"}


def test_empty_index_returns_no_results():
    store = VectorStore(client=FakeClient(FakeCollection()))

    assert store.search([0.1]) == []


@pytest.mark.parametrize(
    ("query_embedding", "top_k"),
    [([], 1), ([0.1], 0)],
)
def test_invalid_search_parameters_raise(query_embedding, top_k):
    store = VectorStore(client=FakeClient(FakeCollection()))

    with pytest.raises(ValueError):
        store.search(query_embedding, top_k=top_k)

def test_upsert_updates_existing_chunk():
    collection = FakeCollection()
    store = VectorStore(client=FakeClient(collection))

    original = make_chunk(
        "id-1", "sample.pdf", "original text", [1], [0.1, 0.2]
    )
    updated = make_chunk(
        "id-1", "sample.pdf", "updated text", [2], [0.3, 0.4]
    )

    store.upsert_chunks([original])
    store.upsert_chunks([updated])

    assert len(collection.records) == 1
    assert collection.records[0][1] == "updated text"