
from pathlib import Path

from pypdf import PdfReader

from src.ingestion.page_mapping import detect_printed_page_label

def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """Extract PDF text page by page, preserving 1-based page numbers."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    reader = PdfReader(str(pdf_path))

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported yet.")

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        page_metadata = detect_printed_page_label(text)

        pages.append({
            "page_number": page_number,
            "text": text,
            **page_metadata,
        })

    if not any(page["text"] for page in pages):
        raise ValueError(
            "No extractable text found. "
            "The PDF may be scanned and require OCR."
        )

    return pages
