import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_loader import load_pdf

pdf_path = PROJECT_ROOT / "data" / "input" / "sample" / "jess107.pdf"

print("Starting PDF extraction...")

pages = load_pdf(pdf_path)

print(f"Total non-empty pages extracted: {len(pages)}")

for page in pages:
    if "Banihal" in page.text or "Baramula" in page.text:
        print(f"\nPhysical page: {page.page_number}")
        print(page.text)

print("\nExtraction check completed.")