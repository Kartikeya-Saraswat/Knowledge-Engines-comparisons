
import re


# Match a standalone Arabic or Roman numeral, optionally near a page edge.
ARABIC_PAGE = re.compile(r"^\s*(\d{1,4})\s*$")
ROMAN_PAGE = re.compile(
    r"^\s*([ivxlcdm]{1,8})\s*$",
    re.IGNORECASE,
)

ROMAN_VALUES = {
    "i": 1, "v": 5, "x": 10, "l": 50,
    "c": 100, "d": 500, "m": 1000,
}


def roman_to_int(value: str) -> int | None:
    """Convert a valid Roman numeral to an integer."""
    value = value.lower()

    total = 0
    previous = 0

    for char in reversed(value):
        current = ROMAN_VALUES[char]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current

    # Reject malformed Roman numerals such as 'iiv' or 'vx'.
    if total <= 0 or int_to_roman(total) != value:
        return None

    return total


def int_to_roman(number: int) -> str:
    """Convert a positive integer to canonical Roman numeral form."""
    if not 1 <= number <= 3999:
        return ""

    values = (
        (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
        (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
        (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
    )

    result = []
    for value, numeral in values:
        count, number = divmod(number, value)
        result.append(numeral * count)

    return "".join(result)


def detect_printed_page_label(text: str) -> dict:
    """
    Detect likely printed page labels near the beginning or end
    of extracted text. Physical PDF page numbers remain authoritative.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    def parse_label(candidate: str) -> dict | None:
        arabic_match = ARABIC_PAGE.fullmatch(candidate)
        if arabic_match:
            value = int(arabic_match.group(1))
            if value > 0:
                return {
                    "page_label": candidate,
                    "printed_page_number": value,
                    "page_label_type": "arabic",
                }

        roman_match = ROMAN_PAGE.fullmatch(candidate)
        if roman_match:
            label = roman_match.group(1)
            value = roman_to_int(label)
            if value is not None:
                return {
                    "page_label": label,
                    "printed_page_number": value,
                    "page_label_type": "roman",
                }

        return None

    # Check the first non-empty line: NCERT PDFs often extract the
    # printed page number before the running header and body text.
    if lines:
        result = parse_label(lines[0])
        if result is not None:
            return result

    # Some pages place the label near the end, before publication footers.
    footer_markers = (
        "reprint ",
        "national council of educational research and training",
    )

    for candidate in reversed(lines[-5:]):
        if any(marker in candidate.lower() for marker in footer_markers):
            continue

        result = parse_label(candidate)
        if result is not None:
            return result

        # Don't search further into ordinary body text.
        break

    return {
        "page_label": None,
        "printed_page_number": None,
        "page_label_type": None,
    }
