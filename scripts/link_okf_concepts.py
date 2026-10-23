
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.okf_linker import (
    find_related_concepts,
    load_concept_catalogue,
    write_concept_index,
    write_related_links,
)
from src.validation.link_validator import validate_markdown_links


SOURCE_NAME = "jess101"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "okf" / SOURCE_NAME


def main():
    print(f"Loading concepts from: {OUTPUT_DIR}")

    catalogue = load_concept_catalogue(OUTPUT_DIR)

    if not catalogue:
        raise RuntimeError(
            "No valid concepts found. Generate OKF concepts first."
        )

    print(f"Loaded {len(catalogue)} concepts.")

    print("Finding related concepts using Groq...")
    relationships = find_related_concepts(
        catalogue,
        max_related=3,
    )

    print("Writing related-concept links...")
    linked_paths = write_related_links(catalogue, relationships)

    print("Generating index.md...")
    index_path = write_concept_index(catalogue, OUTPUT_DIR)

    print("Validating Markdown links...")
    errors = validate_markdown_links(OUTPUT_DIR)

    print(f"Updated {len(linked_paths)} concept files.")
    print(f"Created index: {index_path.name}")

    if errors:
        print(f"\nFound {len(errors)} broken Markdown links:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("All local Markdown links are valid.")
    print("Cross-linking completed successfully.")


if __name__ == "__main__":
    main()
