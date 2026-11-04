
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PageRecord:
    """Text extracted from one physical PDF page."""

    document_name: str
    page_number: int
    text: str


def load_pdf(pdf_path: str | Path) -> list[PageRecord]:
    """Extract non-empty PDF pages while preserving physical page numbers."""

    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path}")

    records = []

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
