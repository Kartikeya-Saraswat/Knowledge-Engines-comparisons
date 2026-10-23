
import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_extractor import extract_pdf_pages


input_dir = Path("data/input/sample")
pdf_files = sorted(input_dir.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in {input_dir}")

for pdf_path in pdf_files:
    print(f"\n{'=' * 60}")
    print(f"File: {pdf_path.name}")

    pages = extract_pdf_pages(pdf_path)
    text_pages = sum(bool(page["text"]) for page in pages)

    print(f"Total PDF pages: {len(pages)}")
    print(f"Pages containing text: {text_pages}")

    for page in pages[:2]:
        print(f"\n--- PDF page {page['page_number']} ---")
        print(page["text"][:300] or "[No extractable text]")

print(f"\nProcessed {len(pdf_files)} PDF files.")
