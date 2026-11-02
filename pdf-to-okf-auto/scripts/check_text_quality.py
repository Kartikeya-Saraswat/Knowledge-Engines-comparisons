import argparse
import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.text_quality import inspect_file

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan an OKF bundle for suspicious text encoding artefacts."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="jess101",
        help="Bundle name or path (default: jess101)",
    )
    args = parser.parse_args()

    source_path = Path(args.source)

    if source_path.exists() and source_path.is_dir():
        bundle_dir = source_path
    else:
        bundle_name = source_path.stem
        bundle_dir = Path("data/output/okf") / bundle_name

    if not bundle_dir.is_dir():
        print(f"ERROR: Bundle directory not found: {bundle_dir}")
        return 2

    markdown_files = sorted(
        path
        for path in bundle_dir.rglob("*.md")
        if path.name.lower() != "index.md"
    )

    if not markdown_files:
        print(f"No concept Markdown files found in {bundle_dir}")
        return 0

    total_findings = 0

    for path in markdown_files:
        findings = inspect_file(path)

        for finding in findings:
            total_findings += 1
            print(
                f"{finding['source']}:{finding['line']}: "
                f"{finding['issue']} — {finding['matched_text']!r}"
            )
            print(f"  {finding['context']}")

    print(
        f"\nScanned {len(markdown_files)} Markdown files; "
        f"found {total_findings} suspicious pattern(s)."
    )

    return 1 if total_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
