
from pathlib import Path

import pytest

from src.ingestion.pdf_loader import load_pdf


def create_test_pdf(pdf_path: Path) -> None:
    """Create a minimal PDF with extractable text."""

    content = (
        b"BT\n"
        b"/F1 12 Tf\n"
        b"72 720 Td\n"
        b"(Hello Hybrid RAG) Tj\n"
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
        (
            b"<< /Length " + str(len(content)).encode() + b" >>\n"
            b"stream\n" + content + b"endstream"
        ),
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


def test_load_pdf_extracts_text_and_metadata(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_test_pdf(pdf_path)

    pages = load_pdf(pdf_path)

    assert len(pages) == 1
    assert pages[0].document_name == "sample.pdf"
    assert pages[0].page_number == 1
    assert "Hello Hybrid RAG" in pages[0].text


def test_load_pdf_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pdf("missing.pdf")


def test_load_pdf_rejects_non_pdf_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a PDF"):
        load_pdf(file_path)
