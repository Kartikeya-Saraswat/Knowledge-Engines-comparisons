"""Measure chunk word and token lengths using the configured FastEmbed model."""

import logging
from pathlib import Path
import sys
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastembed import TextEmbedding

from src.chunking.text_chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_pages,
)
from src.embeddings.embedder import DEFAULT_EMBEDDING_MODEL
from src.ingestion.pdf_loader import load_pdf


PDF_DIRECTORY = PROJECT_ROOT / "data" / "input" / "sample"


def main() -> None:
    logging.disable(logging.CRITICAL)
    model = TextEmbedding(model_name=DEFAULT_EMBEDDING_MODEL)
    tokenizer = model.model.tokenizer
    max_tokens = tokenizer.truncation["max_length"]

    all_word_counts: list[int] = []
    all_token_counts: list[int] = []
    all_cross_page_chunks = 0

    for pdf_path in sorted(PDF_DIRECTORY.rglob("*.pdf")):
        chunks = chunk_pages(
            load_pdf(pdf_path),
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_CHUNK_OVERLAP,
        )
        word_counts = [len(chunk.text.split()) for chunk in chunks]
        token_counts = [
            len(tokenizer.encode(chunk.text).ids)
            for chunk in chunks
        ]
        cross_page_chunks = sum(len(chunk.page_numbers) > 1 for chunk in chunks)
        all_word_counts.extend(word_counts)
        all_token_counts.extend(token_counts)
        all_cross_page_chunks += cross_page_chunks

        if word_counts:
            print(
                f"{pdf_path.name}: chunks={len(chunks)}, "
                f"words median/max={median(word_counts):.0f}/{max(word_counts)}, "
                f"tokens median/max={median(token_counts):.0f}/{max(token_counts)}, "
                f"over model limit={sum(count > max_tokens for count in token_counts)}"
            )

    if not all_word_counts:
        print(f"No PDFs found in {PDF_DIRECTORY}")
        return

    print(
        f"ALL: chunks={len(all_word_counts)}, "
        f"words median/max={median(all_word_counts):.0f}/{max(all_word_counts)}, "
        f"tokens median/max={median(all_token_counts):.0f}/{max(all_token_counts)}, "
        f"over model limit={sum(count > max_tokens for count in all_token_counts)}, "
        f"cross-page chunks={all_cross_page_chunks}"
    )


if __name__ == "__main__":
    main()
