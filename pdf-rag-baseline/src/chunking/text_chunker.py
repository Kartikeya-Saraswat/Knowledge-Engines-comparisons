"""Create overlapping word chunks while keeping their PDF page provenance."""

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from src.ingestion.pdf_loader import PageRecord


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


@dataclass(frozen=True)
class ChunkRecord:
    """A chunk of document text and the physical pages it came from."""

    chunk_id: str
    document_name: str
    text: str
    page_numbers: tuple[int, ...]


def chunk_pages(
    pages: Iterable[PageRecord],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkRecord]:
    """Split page records into overlapping word chunks.

    Chunk boundaries are based on whitespace-separated words. The page number
    for every word is carried through chunking, so chunks spanning PDF pages
    retain all contributing physical page numbers in document order.
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

    words: list[str] = []
    word_pages: list[int] = []
    for page in page_records:
        page_words = page.text.split()
        words.extend(page_words)
        word_pages.extend([page.page_number] * len(page_words))

    chunks: list[ChunkRecord] = []
    step = chunk_size - overlap
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        page_numbers = tuple(dict.fromkeys(word_pages[start:end]))
        chunk_index = len(chunks)
        identity = (
            f"{document_name}\0{chunk_index}\0"
            f"{chunk_size}\0{overlap}\0{chunk_text}"
        )        
        chunk_id = sha256(identity.encode("utf-8")).hexdigest()
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
        start += step

    return chunks
