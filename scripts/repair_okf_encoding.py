
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.okf_linker import (
    load_concept_catalogue,
    write_concept_index,
)
from src.ingestion.text_cleaner import fix_encoding_artifacts
from src.validation.link_validator import validate_markdown_links


OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "okf" / "jess101"


def main():
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"OKF directory not found: {OUTPUT_DIR}"
        )

    updated = 0

    # Change encoding artefacts only; preserve Markdown structure.
    for path in OUTPUT_DIR.glob("*.md"):
        original = path.read_text(encoding="utf-8")
        corrected = fix_encoding_artifacts(original)

        if corrected != original:
            path.write_text(corrected, encoding="utf-8")
            updated += 1

    # Rebuild the index from the corrected concept metadata.
    catalogue = load_concept_catalogue(OUTPUT_DIR)
    index_path = write_concept_index(catalogue, OUTPUT_DIR)

    errors = validate_markdown_links(OUTPUT_DIR)

    print(f"Repaired encoding in {updated} Markdown files.")
    print(f"Regenerated index: {index_path.name}")
    print(f"Loaded {len(catalogue)} concepts.")

    if errors:
        print(f"Found {len(errors)} broken links:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("All local Markdown links are valid.")


if __name__ == "__main__":
    main()
