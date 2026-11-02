
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_extractor import extract_pdf_pages
from src.ingestion.save_extraction import save_extracted_pages

input_dir = PROJECT_ROOT / "data" / "input" / "sample"
output_dir = PROJECT_ROOT / "data" / "output" / "extracted"

pdf_files = sorted(input_dir.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(f"No PDFs found in {input_dir}")

for pdf_path in pdf_files:
    pages = extract_pdf_pages(pdf_path)
    output_path = save_extracted_pages(pdf_path, pages, output_dir)

    print(
        f"Saved {pdf_path.name}: "
        f"{len(pages)} pages -> {output_path.name}"
    )

print(f"\nCompleted extraction for {len(pdf_files)} PDFs.")
