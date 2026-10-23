
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.provenance_validator import (
    validate_provenance_directory,
)

OKF_DIR = PROJECT_ROOT / "data" / "output" / "okf" / "jess101"
EXTRACTION_FILE = (
    PROJECT_ROOT / "data" / "output" / "cleaned" / "jess101.json"
)


def main():
    results = validate_provenance_directory(
        OKF_DIR,
        EXTRACTION_FILE,
    )

    failed = {
        filename: errors
        for filename, errors in results.items()
        if errors
    }

    print(f"Files checked: {len(results)}")
    print(f"Passed: {len(results) - len(failed)}")
    print(f"Failed: {len(failed)}")

    for filename, errors in failed.items():
        print(f"\n{filename}")
        for error in errors:
            print(f"  - {error}")

    if failed:
        raise SystemExit(1)

    print("\nAll provenance checks passed.")


if __name__ == "__main__":
    main()
