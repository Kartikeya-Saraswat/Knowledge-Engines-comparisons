
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.okf_validator import validate_okf_directory

TARGET_DIR = PROJECT_ROOT / "data" / "output" / "okf" / "jess101"


def main():
    results = validate_okf_directory(TARGET_DIR)
    total = len(results)
    failed = {
        filename: errors
        for filename, errors in results.items()
        if errors
    }

    print(f"Files checked: {total}")
    print(f"Passed: {total - len(failed)}")
    print(f"Failed: {len(failed)}")

    for filename, errors in failed.items():
        print(f"\n{filename}")
        for error in errors:
            print(f"  - {error}")

    if failed:
        raise SystemExit(1)

    print("\nAll OKF files passed validation.")


if __name__ == "__main__":
    main()
