import pytest

from src.ingestion.pdf_loader import PageRecord
from src.chunking.text_chunker import (
    chunk_pages,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


def test_chunker_returns_empty_list_for_empty_input():
    assert chunk_pages([]) == []


def test_chunker_preserves_document_and_page_provenance():
    pages = [
        PageRecord(
            document_name="sample.pdf",
            page_number=1,
            text="This is the first page.",
        ),
        PageRecord(
            document_name="sample.pdf",
            page_number=2,
            text="This is the second page.",
        ),
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) == 1
    assert chunks[0].document_name == "sample.pdf"
    assert chunks[0].page_numbers == (1, 2)


def test_chunker_splits_long_text():
    text = " ".join(f"word{i}" for i in range(600))

    pages = [
        PageRecord(
            document_name="sample.pdf",
            page_number=1,
            text=text,
        )
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) > 1
    assert all(
        len(chunk.text.split()) <= DEFAULT_CHUNK_SIZE
        for chunk in chunks
    )


def test_chunker_preserves_overlap():
    text = " ".join(f"word{i}" for i in range(600))

    pages = [
        PageRecord(
            document_name="sample.pdf",
            page_number=1,
            text=text,
        )
    ]

    chunks = chunk_pages(pages)

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()

    assert first_words[-DEFAULT_CHUNK_OVERLAP:] == (
        second_words[:DEFAULT_CHUNK_OVERLAP]
    )


def test_chunk_ids_are_deterministic():
    pages = [
        PageRecord(
            document_name="sample.pdf",
            page_number=1,
            text="A sample paragraph for testing.",
        )
    ]

    first_run = chunk_pages(pages)
    second_run = chunk_pages(pages)

    assert first_run[0].chunk_id == second_run[0].chunk_id


def test_chunker_rejects_invalid_parameters():
    pages = [
        PageRecord(
            document_name="sample.pdf",
            page_number=1,
            text="Sample text.",
        )
    ]

    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=0)

    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=100, chunk_overlap=100)


def test_chunker_rejects_multiple_documents():
    pages = [
        PageRecord("first.pdf", 1, "First document."),
        PageRecord("second.pdf", 1, "Second document."),
    ]

    with pytest.raises(ValueError):
        chunk_pages(pages)