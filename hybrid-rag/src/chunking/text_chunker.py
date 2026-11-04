import hashlib
import re
from dataclasses import dataclass

from src.ingestion.pdf_loader import PageRecord


DEFAULT_CHUNK_SIZE = 275
DEFAULT_CHUNK_OVERLAP = 50


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_name: str
    text: str
    page_numbers: tuple[int, ...]


def _extract_paragraphs(text: str) -> list[str]:
    """Split page text into paragraphs while joining wrapped lines."""
    paragraphs = []
    current_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            if current_lines:
                paragraphs.append(" ".join(current_lines))
                current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        paragraphs.append(" ".join(current_lines))

    return paragraphs


def chunk_pages(
    pages: list[PageRecord],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkRecord]:
    """Create paragraph-aware chunks while preserving page provenance."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

    if not pages:
        return []

    document_names = {page.document_name for page in pages}
    if len(document_names) != 1:
        raise ValueError("All pages must belong to the same document")

    document_name = pages[0].document_name

    # Flatten words while retaining their original page numbers.
    words = []
    word_pages = []
    paragraph_starts = set()

    for page in pages:
        for paragraph in _extract_paragraphs(page.text):
            paragraph_words = paragraph.split()

            if not paragraph_words:
                continue

            paragraph_starts.add(len(words))
            words.extend(paragraph_words)
            word_pages.extend([page.page_number] * len(paragraph_words))

    if not words:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        max_end = min(start + chunk_size, len(words))
        end = max_end

        # Prefer a paragraph boundary when one is available
        # in the latter half of the chunk.
        if max_end < len(words):
            minimum_boundary = start + chunk_size // 2

            candidates = [
                position
                for position in paragraph_starts
                if minimum_boundary <= position < max_end
            ]

            if candidates:
                end = max(candidates)

        chunk_words = words[start:end]

        # Restore paragraph breaks within the selected chunk.
        text_parts = []
        for position in range(start, end):
            if position > start and position in paragraph_starts:
                text_parts.append("\n\n")
            elif position > start:
                text_parts.append(" ")

            text_parts.append(words[position])

        chunk_text = "".join(text_parts)

        page_numbers = tuple(
            dict.fromkeys(word_pages[start:end])
        )

        identity = (
            f"{document_name}|{chunk_index}|"
            f"{chunk_size}|{chunk_overlap}|{chunk_text}"
        )
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()

        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_name=document_name,
                text=chunk_text,
                page_numbers=page_numbers,
            )
        )

        if end == len(words):
            break

        # Start the next chunk with overlapping context.
        next_start = max(start + 1, end - chunk_overlap)

        start = next_start
        chunk_index += 1

    return chunks