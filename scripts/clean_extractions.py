
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.clean_extractions import clean_extraction_file

input_dir = PROJECT_ROOT / "data" / "output" / "extracted"
output_dir = PROJECT_ROOT / "data" / "output" / "cleaned"

json_files = sorted(input_dir.glob("*.json"))

if not json_files:
    raise FileNotFoundError(f"No JSON files found in {input_dir}")

for json_path in json_files:
    output_path = clean_extraction_file(json_path, output_dir)
    print(f"Cleaned {json_path.name} -> {output_path.name}")

print(f"\nCompleted cleaning for {len(json_files)} JSON files.")
