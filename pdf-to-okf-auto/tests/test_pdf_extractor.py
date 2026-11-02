
from pathlib import Path

import pytest

from src.ingestion.pdf_extractor import extract_pdf_pages


def create_test_pdf(pdf_path: Path) -> None:
    """Create a minimal one-page PDF containing extractable text."""
    content = (
        b"BT\n"
        b"/F1 12 Tf\n"
        b"72 720 Td\n"
        b"(Hello NCERT) Tj\n"
        b"ET\n"
    )

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> "
            b"/Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\n"
        b"stream\n" + content + b"endstream",
    ]

    with pdf_path.open("wb") as pdf:
        pdf.write(b"%PDF-1.4\n")
        offsets = [0]

        for number, obj in enumerate(objects, start=1):
            offsets.append(pdf.tell())
            pdf.write(f"{number} 0 obj\n".encode())
            pdf.write(obj + b"\nendobj\n")

        xref_offset = pdf.tell()
        pdf.write(f"xref\n0 {len(objects) + 1}\n".encode())
        pdf.write(b"0000000000 65535 f \n")

        for offset in offsets[1:]:
            pdf.write(f"{offset:010d} 00000 n \n".encode())

        pdf.write(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n".encode()
        )


def test_extract_pdf_pages_returns_page_numbers(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    create_test_pdf(pdf_path)

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "Hello NCERT" in pages[0]["text"]


def test_missing_pdf_raises_error():
    with pytest.raises(FileNotFoundError):
        extract_pdf_pages("does_not_exist.pdf")


def test_non_pdf_file_raises_error(tmp_path: Path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a PDF"):
        extract_pdf_pages(text_file)
