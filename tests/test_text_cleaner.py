
import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.text_cleaner import clean_extracted_text


def test_basic_text():
    assert clean_extracted_text("hello world") == "hello world"


def test_encoding_cleanup():
    sample = "publisherâ€™s Â© â€“ â€¢"
    result = clean_extracted_text(sample)

    print("Input: ", repr(sample))
    print("Output:", repr(result))

    assert "publisher’s" in result
    assert "©" in result
    assert "–" in result
    assert "•" in result


if __name__ == "__main__":
    test_basic_text()
    test_encoding_cleanup()
    print("All text-cleaner checks passed!")
