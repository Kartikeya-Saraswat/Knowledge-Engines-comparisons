
from pathlib import Path

import pymupdf

from src.ingestion.page_mapping import detect_printed_page_label


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """Extract text and page metadata from a PDF using PyMuPDF."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    pages = []

    try:
        with pymupdf.open(pdf_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                text = (page.get_text("text") or "").strip()
                page_metadata = detect_printed_page_label(text)

                pages.append({
                    "page_number": page_number,
                    "text": text,
                    **page_metadata,
                })

    except pymupdf.FileDataError as exc:
        raise ValueError(f"Unable to read PDF: {pdf_path}") from exc

    return pages
