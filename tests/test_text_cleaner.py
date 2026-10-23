from src.ingestion.text_cleaner import fix_encoding_artifacts

from src.ingestion.text_cleaner import clean_extracted_text


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

def test_cleaner_fixes_common_pdf_encoding_artefacts():
    text = (
        "setâ€‘up\n"
        "soil erosion â€“ causes\n"
        "Indiaâ€™s resources\n"
        "moistureâ€‘retentive"
    )

    cleaned = clean_extracted_text(text)

    assert "set-up" in cleaned
    assert "soil erosion – causes" in cleaned
    assert "India’s resources" in cleaned
    assert "moisture-retentive" in cleaned


def test_cleaner_preserves_normal_unicode():
    text = "Sustainable development — today’s needs…"

    assert clean_extracted_text(text) == text


def test_cleaner_normalizes_whitespace():
    text = "First line\r\n\r\n\r\nSecond line"

    assert clean_extracted_text(text) == "First line\n\nSecond line"

def test_encoding_fixes_preserve_whitespace_and_markdown():
    text = (
        "---\n"
        "description: Indiaâ€™s resources\n"
        "---\n\n"
        "Text with setâ€‘up and an en dash â€“ here.\n\n"
        "## Related concepts\n\n"
        "- [Planning](planning.md)\n"
    )

    cleaned = fix_encoding_artifacts(text)

    assert "India’s resources" in cleaned
    assert "set-up" in cleaned
    assert "en dash – here" in cleaned
    assert "- [Planning](planning.md)" in cleaned
    assert cleaned.count("\n\n") == text.count("\n\n")