"""Use the repository shared PDF chunker."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from retrieval_shared.pdf_chunking import ChunkRecord, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from retrieval_shared.pdf_chunking import chunk_pages as _shared_chunk_pages

def chunk_pages(pages, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    return _shared_chunk_pages(pages, chunk_size=chunk_size, overlap=chunk_overlap)
