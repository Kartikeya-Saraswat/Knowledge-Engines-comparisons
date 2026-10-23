
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.okf_generator import (
    generate_concepts,
    write_okf_concepts,
)


INPUT_DIR = PROJECT_ROOT / "data" / "output" / "cleaned"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "okf"
BATCH_SIZE = 3


def normalize_source_name(value: str) -> str:
    """Accept a bundle name, PDF name, or JSON name."""
    return Path(value).stem


def main():
    parser = argparse.ArgumentParser(
        description="Generate OKF concepts from a cleaned PDF extraction."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="jess101",
        help="Source bundle, PDF, or JSON name (default: jess101)",
    )
    args = parser.parse_args()

    source_name = normalize_source_name(args.source)
    input_path = INPUT_DIR / f"{source_name}.json"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned extraction not found: {input_path}"
        )

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    source_file = payload["source_file"]
    pages = payload["pages"]

    document_output_dir = OUTPUT_DIR / Path(source_file).stem
    total_written = 0

    for start in range(0, len(pages), BATCH_SIZE):
        batch = pages[start:start + BATCH_SIZE]
        concepts = generate_concepts(batch, source_file)

        paths = write_okf_concepts(
            concepts,
            source_file,
            document_output_dir,
            start_index=total_written + 1,
        )

        total_written += len(paths)

        page_numbers = [
            page["page_number"] for page in batch
        ]

        print(
            f"Pages {page_numbers}: "
            f"generated {len(concepts)} concepts, "
            f"wrote {len(paths)} files"
        )

    print(f"\nCompleted {source_file}")
    print(f"Total OKF files written: {total_written}")
    print(f"Output directory: {document_output_dir}")


if __name__ == "__main__":
    main()
