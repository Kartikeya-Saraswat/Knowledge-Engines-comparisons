"""Load text from PDFs while retaining source document and physical page data."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PageRecord:
    """Text extracted from one non-empty physical PDF page."""

    document_name: str
    page_number: int
    text: str


def load_pdf(pdf_path: str | Path) -> list[PageRecord]:
    """Extract non-empty pages from ``pdf_path`` in document order.

    Page numbers are one-based physical PDF page numbers, including gaps where
    pages were skipped because they contained no extractable text.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path}")

    records: list[PageRecord] = []
    with path.open("rb") as pdf_file:
        document = PdfReader(pdf_file)
        for page_index, page in enumerate(document.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            records.append(
                PageRecord(
                    document_name=path.name,
                    page_number=page_index + 1,
                    text=text,
                )
            )

    return records
