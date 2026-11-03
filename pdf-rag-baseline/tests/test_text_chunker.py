import pytest

from src.chunking.text_chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    _is_heading,
    ChunkRecord,
    chunk_pages,
)
from src.ingestion.pdf_loader import PageRecord


def test_default_chunk_settings():
    assert DEFAULT_CHUNK_SIZE == 275
    assert DEFAULT_CHUNK_OVERLAP == 50


def test_wrap_lines_are_joined_and_paragraph_breaks_are_preserved():
    page = PageRecord(
        "sample.pdf",
        1,
        "First paragraph wraps\nonto another line and ends here.\n"
        "Second paragraph stays whole.",
    )

    chunks = chunk_pages([page], chunk_size=20, overlap=2)

    assert len(chunks) == 1
    assert chunks[0].text == (
        "First paragraph wraps onto another line and ends here.\n\n"
        "Second paragraph stays whole."
    )


def test_chunk_size_and_overlap_split_oversized_paragraph():
    words = [f"word{i}" for i in range(10)]
    chunks = chunk_pages(
        [PageRecord("sample.pdf", 1, " ".join(words))],
        chunk_size=6,
        overlap=2,
    )

    assert [chunk.text.split() for chunk in chunks] == [words[:6], words[4:]]


def test_cross_page_chunk_preserves_all_page_provenance():
    pages = [
        PageRecord("sample.pdf", 2, "one two three"),
        PageRecord("sample.pdf", 5, "four five six"),
    ]

    chunks = chunk_pages(pages, chunk_size=6, overlap=1)

    assert [chunk.text for chunk in chunks] == ["one two three\n\nfour five six"]
    assert chunks[0].page_numbers == (2, 5)


def test_chunks_do_not_cross_detected_section_headings():
    page = PageRecord(
        "sample.pdf",
        1,
        "First section has some content.\nSECOND SECTION\n"
        "Second section has different content.",
    )

    chunks = chunk_pages([page], chunk_size=50, overlap=5)

    assert len(chunks) == 2
    assert "SECOND SECTION" not in chunks[0].text
    assert chunks[1].text.startswith("SECOND SECTION")


def test_chunk_ids_are_deterministic_and_unique():
    pages = [PageRecord("sample.pdf", 1, "repeat repeat repeat repeat repeat")]

    first_run = chunk_pages(pages, chunk_size=2, overlap=1)
    second_run = chunk_pages(pages, chunk_size=2, overlap=1)

    assert [chunk.chunk_id for chunk in first_run] == [
        chunk.chunk_id for chunk in second_run
    ]
    assert len({chunk.chunk_id for chunk in first_run}) == len(first_run)


def test_chunk_records_preserve_document_and_page_metadata():
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

@pytest.mark.parametrize(
    "heading",
    [
        "Activity",
        "Activities",
        "Exercise",
        "Exercises",
        "Quiz Drive",
        "Summary",
        "Case Study",
        "Answer Key",
    ],
)
def test_recognises_callout_headings(heading):
    assert _is_heading(heading)


def test_does_not_recognise_normal_sentence_as_heading():
    assert not _is_heading(
        "The railway line connects Banihal and Baramula."
    )

def test_callout_heading_starts_new_chunk_section():
    pages = [
        PageRecord(
            document_name="jess107.pdf",
            page_number=6,
            text=(
                "The railway line was extended.\n\n"
                "Activity\n"
                "Locate Banihal and Baramula on the map of India."
            ),
        )
    ]

    chunks = chunk_pages(pages, chunk_size=100, overlap=10)

    assert len(chunks) == 2
    assert "The railway line was extended." in chunks[0].text
    assert "Activity" in chunks[1].text
    assert "Locate Banihal and Baramula" in chunks[1].text