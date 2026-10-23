
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.semantic_quality import analyze_semantic_quality

OKF_DIR = PROJECT_ROOT / "data" / "output" / "okf" / "jess101"
EXTRACTION_FILE = (
    PROJECT_ROOT / "data" / "output" / "cleaned" / "jess101.json"
)
REPORT_FILE = PROJECT_ROOT / "data" / "output" / "semantic_quality_jess101.json"


def main():
    report = analyze_semantic_quality(OKF_DIR, EXTRACTION_FILE)

    print(f"Concepts analyzed: {report['concept_count']}")
    print(f"Low-overlap flags: {len(report['grounding_flags'])}")
    print(f"Potential duplicate pairs: {len(report['duplicate_candidates'])}")

    if report["grounding_flags"]:
        print("\nConcepts to review:")
        for item in report["grounding_flags"]:
            print(
                f"  {item['file']}: overlap={item['overlap']}, "
                f"pages={item['pages']}"
            )

    if report["duplicate_candidates"]:
        print("\nPotential duplicates:")
        for item in report["duplicate_candidates"]:
            print(
                f"  {item['file_a']} <-> {item['file_b']}: "
                f"similarity={item['similarity']}"
            )

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nFull report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
