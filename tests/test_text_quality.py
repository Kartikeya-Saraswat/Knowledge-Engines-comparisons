
from src.validation.text_quality import inspect_text


def test_detects_mojibake_with_line_number():
    findings = inspect_text("Normal text\nIndiaâ€™s climate", "sample.md")

    assert len(findings) == 1
    assert findings[0]["source"] == "sample.md"
    assert findings[0]["line"] == 2
    assert findings[0]["issue"] == "mojibake_apostrophe"
    assert findings[0]["matched_text"] == "â€™"


def test_detects_replacement_character():
    findings = inspect_text("Some damaged text � here")

    assert len(findings) == 1
    assert findings[0]["issue"] == "replacement_character"


def test_clean_text_has_no_findings():
    findings = inspect_text("India’s climate — and its rainfall…")

    assert findings == []


def test_reports_multiple_issues():
    findings = inspect_text("Indiaâ€™s soil â€“ erosion")

    assert len(findings) == 2
    assert {item["issue"] for item in findings} == {
        "mojibake_apostrophe",
        "mojibake_dashes",
    }


def test_includes_context():
    findings = inspect_text("The setâ€‘up failed", "document.md")

    assert findings[0]["context"] == "The setâ€‘up failed"
