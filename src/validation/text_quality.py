
import re
from pathlib import Path


# Patterns that often indicate text encoding or PDF extraction problems.
SUSPICIOUS_PATTERNS = {
    "replacement_character": re.compile("\uFFFD"),
    "mojibake_apostrophe": re.compile(r"â€™|â€˜"),
    "mojibake_quotes": re.compile(r"â€œ|â€\x9d"),
    "mojibake_dashes": re.compile(r"â€“|â€”|â€‘"),
    "mojibake_ellipsis_or_bullet": re.compile(r"â€¦|â€¢"),
    "unexpected_copyright_symbols": re.compile(r"Â©|Â®"),
    "missing_space_after_comma": re.compile(r",(?=[A-Za-z])"),
}


def inspect_text(text: str, source: str = "<text>") -> list[dict]:
    """Report suspicious text patterns with line numbers; do not alter text."""
    findings = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        for issue, pattern in SUSPICIOUS_PATTERNS.items():
            for match in pattern.finditer(line):
                findings.append(
                    {
                        "source": source,
                        "line": line_number,
                        "issue": issue,
                        "matched_text": match.group(),
                        "context": line.strip()[:200],
                    }
                )

    return findings


def inspect_file(path: str | Path) -> list[dict]:
    """Inspect a UTF-8 text file and report suspicious patterns."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return inspect_text(text, source=str(file_path))
