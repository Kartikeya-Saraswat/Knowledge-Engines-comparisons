import pytest

from src.ingestion import pdf_loader
from src.ingestion.pdf_loader import PageRecord, load_pdf


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakeReader:
    def __init__(self, pages):
        self.pages = [FakePage(text) for text in pages]


def patch_reader(monkeypatch, pages):
    monkeypatch.setattr(pdf_loader, "PdfReader", lambda _stream: FakeReader(pages))


def test_load_pdf_preserves_physical_page_numbers(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"placeholder")
    patch_reader(monkeypatch, ["This is page one.", None, "This is page three."])

    records = load_pdf(pdf_path)

    assert records == [
        PageRecord("sample.pdf", 1, "This is page one."),
        PageRecord("sample.pdf", 3, "This is page three."),
    ]


def test_load_pdf_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pdf("missing.pdf")


def test_load_pdf_invalid_extension(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Not a PDF")

    with pytest.raises(ValueError):
        load_pdf(file_path)


def test_load_pdf_empty_document(tmp_path, monkeypatch):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"placeholder")
    patch_reader(monkeypatch, [None, "", "  "])

    assert load_pdf(pdf_path) == []

def test_load_pdf_strips_page_text(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"placeholder")
    patch_reader(monkeypatch, ["  Hello PDF  \n"])

    records = load_pdf(pdf_path)

    assert records[0].text == "Hello PDF"