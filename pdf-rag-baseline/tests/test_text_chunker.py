import pytest

from src.chunking.text_chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    ChunkRecord,
    chunk_pages,
)
from src.ingestion.pdf_loader import PageRecord


def test_default_chunk_settings():
    assert DEFAULT_CHUNK_SIZE == 800
    assert DEFAULT_CHUNK_OVERLAP == 150


def test_chunk_size_and_overlap():
    words = [f"word{i}" for i in range(10)]
    chunks = chunk_pages(
        [PageRecord("sample.pdf", 1, " ".join(words))],
        chunk_size=6,
        overlap=2,
    )

    assert [chunk.text.split() for chunk in chunks] == [
        words[:6],
        words[4:],
    ]


def test_chunk_provenance_includes_all_cross_page_sources():
    pages = [
        PageRecord("sample.pdf", 2, "one two three"),
        PageRecord("sample.pdf", 5, "four five six"),
    ]

    chunks = chunk_pages(pages, chunk_size=4, overlap=1)

    assert chunks[0].text == "one two three four"
    assert chunks[0].page_numbers == (2, 5)
    assert chunks[1].text == "four five six"
    assert chunks[1].page_numbers == (5,)


def test_chunk_ids_are_deterministic_and_unique():
    pages = [PageRecord("sample.pdf", 1, "repeat repeat repeat repeat repeat")]

    first_run = chunk_pages(pages, chunk_size=2, overlap=1)
    second_run = chunk_pages(pages, chunk_size=2, overlap=1)

    assert [chunk.chunk_id for chunk in first_run] == [
        chunk.chunk_id for chunk in second_run
    ]
    assert len({chunk.chunk_id for chunk in first_run}) == len(first_run)


def test_chunk_records_preserve_document_name():
    chunks = chunk_pages(
        [PageRecord("report.pdf", 7, "hello world")], chunk_size=2, overlap=0
    )

    assert chunks == [
        ChunkRecord(
            chunk_id=chunks[0].chunk_id,
            document_name="report.pdf",
            text="hello world",
            page_numbers=(7,),
        )
    ]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (2, -1), (2, 2)],
)
def test_invalid_chunk_settings_raise(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=chunk_size, overlap=overlap)


def test_mixed_documents_are_rejected():
    pages = [
        PageRecord("first.pdf", 1, "one"),
        PageRecord("second.pdf", 1, "two"),
    ]

    with pytest.raises(ValueError, match="same document"):
        chunk_pages(pages)
