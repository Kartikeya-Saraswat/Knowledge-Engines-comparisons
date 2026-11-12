"""Build paragraph-aware chunks while retaining PDF page provenance."""

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Iterable

from typing import Protocol

class PageRecord(Protocol):
    document_name: str
    page_number: int
    text: str


DEFAULT_CHUNK_SIZE = 275
DEFAULT_CHUNK_OVERLAP = 50

_LIST_ITEM = re.compile(
    r"^(?:[\u2022\u25aa\u25e6\u25cf*-]|\ufffd|\(?[a-zA-Z0-9]+\))\s+"
)
_SECTION_LABEL = re.compile(r"^(?:chapter|unit|part|section|appendix)\b", re.I)
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*\s+[A-Z]")
_CALLOUT_HEADING = re.compile(
    r"^(?:activity|activities|exercise|exercises|"
    r"quiz(?:\s+drive)?|multiple choice questions|"
    r"answer key|summary|case study)\s*[:.]?$",
    re.IGNORECASE,
    )
_PAGE_NUMBER = re.compile(r"^\d{1,4}$")


@dataclass(frozen=True)
class ChunkRecord:
    """A text chunk and the physical PDF page it came from."""

    chunk_id: str
    document_name: str
    text: str
    page_numbers: tuple[int, ...]


@dataclass(frozen=True)
class _Paragraph:
    text: str
    page_number: int
    is_heading: bool = False


def chunk_pages(
    pages: Iterable[PageRecord],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkRecord]:
    """Chunk page text around paragraph and section boundaries.

    Wrapped PDF lines are joined into paragraphs using sentence endings,
    headings, and list markers. Chunks may span pages when paragraph flow
    continues, but detected section headings are hard boundaries. Unusually
    long paragraphs are split at word boundaries. Overlap starts as close as
    possible to the requested word count while preferring paragraph boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    page_records = list(pages)
    if not page_records:
        return []

    document_names = {page.document_name for page in page_records}
    if len(document_names) != 1:
        raise ValueError("All page records must belong to the same document")
    document_name = page_records[0].document_name

    page_paragraphs = [
        (page, _extract_paragraphs(page.text, page.page_number))
        for page in page_records
    ]
    repeated_headers = _find_repeated_headers(page_paragraphs)

    document_paragraphs: list[_Paragraph] = []
    for _, paragraphs in page_paragraphs:
        document_paragraphs.extend(
            paragraph
            for paragraph in paragraphs
            if not (
                paragraph.is_heading
                and _normalize(paragraph.text) in repeated_headers
            )
        )

    chunks: list[ChunkRecord] = []
    for section in _group_sections(document_paragraphs):
        chunks.extend(
            _chunk_section(
                section,
                document_name=document_name,
                chunk_size=chunk_size,
                overlap=overlap,
                first_chunk_index=len(chunks),
            )
        )

    return chunks


def _extract_paragraphs(text: str, page_number: int) -> list[_Paragraph]:
    paragraphs: list[_Paragraph] = []
    current_lines: list[str] = []

    def flush() -> None:
        if current_lines:
            paragraph_text = " ".join(current_lines).strip()
            if paragraph_text:
                paragraphs.append(_Paragraph(paragraph_text, page_number))
            current_lines.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or _PAGE_NUMBER.fullmatch(line):
            flush()
            continue

        if _is_heading(line):
            flush()
            paragraphs.append(_Paragraph(line, page_number, is_heading=True))
            continue

        if _LIST_ITEM.match(line):
            flush()

        current_lines.append(line)
        if _ends_sentence(line):
            flush()

    flush()
    return paragraphs


def _is_heading(line: str) -> bool:
    if (
        _SECTION_LABEL.match(line)
        or _NUMBERED_HEADING.match(line)
        or _CALLOUT_HEADING.match(line)
    ):
        return True

    if len(line.split()) > 12:
        return False

    letters = [character for character in line if character.isalpha()]

    return (
        len(letters) >= 3
        and sum(char.isupper() for char in letters) / len(letters) >= 0.8
    )


def _ends_sentence(line: str) -> bool:
    return line.endswith((".", "?", "!", ".\u201d", "?\u201d", "!\u201d", ".\u2019", "?\u2019", "!\u2019"))


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _find_repeated_headers(
    page_paragraphs: list[tuple[PageRecord, list[_Paragraph]]],
) -> set[str]:
    header_counts: dict[str, int] = {}
    for _, paragraphs in page_paragraphs:
        seen_on_page = {
            _normalize(paragraph.text)
            for paragraph in paragraphs
            if paragraph.is_heading
        }
        for header in seen_on_page:
            header_counts[header] = header_counts.get(header, 0) + 1
    return {header for header, count in header_counts.items() if count > 1}


def _group_sections(paragraphs: list[_Paragraph]) -> list[list[_Paragraph]]:
    sections: list[list[_Paragraph]] = []
    current: list[_Paragraph] = []
    for paragraph in paragraphs:
        if paragraph.is_heading and current:
            sections.append(current)
            current = []
        current.append(paragraph)
    if current:
        sections.append(current)
    return sections


def _chunk_section(
    paragraphs: list[_Paragraph],
    *,
    document_name: str,
    chunk_size: int,
    overlap: int,
    first_chunk_index: int,
) -> list[ChunkRecord]:
    words: list[str] = []
    paragraph_ids: list[int] = []
    word_pages: list[int] = []
    paragraph_starts: list[int] = []
    paragraph_ends: list[int] = []

    for paragraph_index, paragraph in enumerate(paragraphs):
        paragraph_words = paragraph.text.split()
        if not paragraph_words:
            continue
        paragraph_starts.append(len(words))
        words.extend(paragraph_words)
        paragraph_ids.extend([paragraph_index] * len(paragraph_words))
        word_pages.extend([paragraph.page_number] * len(paragraph_words))
        paragraph_ends.append(len(words))

    if not words:
        return []

    chunks: list[ChunkRecord] = []
    start = 0
    while start < len(words):
        max_end = min(start + chunk_size, len(words))
        possible_ends = [
            boundary for boundary in paragraph_ends if start < boundary <= max_end
        ]
        end = max(possible_ends) if possible_ends else max_end
        chunk_text = _join_paragraph_words(words, paragraph_ids, start, end)
        chunk_index = first_chunk_index + len(chunks)
        identity = (
            f"{document_name}\0{chunk_index}\0"
            f"{chunk_size}\0{overlap}\0{chunk_text}"
        )
        chunks.append(
            ChunkRecord(
                chunk_id=sha256(identity.encode("utf-8")).hexdigest(),
                document_name=document_name,
                text=chunk_text,
                page_numbers=tuple(dict.fromkeys(word_pages[start:end])),
            )
        )

        if end == len(words):
            break

        target_start = max(start + 1, end - overlap)
        possible_starts = [
            boundary for boundary in paragraph_starts if start < boundary <= end
        ]
        start = (
            min(
                possible_starts,
                key=lambda boundary: (abs(boundary - target_start), -boundary),
            )
            if possible_starts
            else target_start
        )

    return chunks


def _join_paragraph_words(
    words: list[str], paragraph_ids: list[int], start: int, end: int
) -> str:
    pieces: list[str] = []
    current_paragraph_id = paragraph_ids[start]
    current_words: list[str] = []
    for index in range(start, end):
        paragraph_id = paragraph_ids[index]
        if paragraph_id != current_paragraph_id:
            pieces.append(" ".join(current_words))
            current_words = []
            current_paragraph_id = paragraph_id
        current_words.append(words[index])
    pieces.append(" ".join(current_words))
    return "\n\n".join(pieces)
