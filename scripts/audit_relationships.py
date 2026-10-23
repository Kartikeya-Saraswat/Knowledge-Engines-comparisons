
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.okf_linker import load_concept_catalogue
from src.validation.relationship_audit import audit_relationships


OUTPUT_ROOT = PROJECT_ROOT / "data" / "output" / "okf"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report OKF concept links missing reverse links."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="jess101",
        help="Bundle name or directory (default: jess101)",
    )
    args = parser.parse_args()

    supplied_path = Path(args.source)

    if supplied_path.is_dir():
        bundle_dir = supplied_path
    else:
        bundle_dir = OUTPUT_ROOT / supplied_path.stem

    if not bundle_dir.is_dir():
        print(f"ERROR: Bundle directory not found: {bundle_dir}")
        return 2

    catalogue = load_concept_catalogue(bundle_dir)
    findings = audit_relationships(catalogue)

    print(f"Bundle: {bundle_dir.name}")
    print(f"Concepts analysed: {len(catalogue)}")
    print(f"Missing reverse links: {len(findings)}")

    titles_by_id = {
        concept["id"]: concept["title"]
        for concept in catalogue
    }

    for finding in findings:
        source = titles_by_id[finding["source_id"]]
        target = titles_by_id[finding["target_id"]]
        print(f"- {source} -> {target}")

    if findings:
        print(
            "\nThese are review suggestions, not necessarily errors. "
            "The audit did not modify any files."
        )
    else:
        print("No missing reverse links found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
