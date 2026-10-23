
def clean_extracted_text(text: str) -> str:
    """Normalize common PDF text encoding artefacts and whitespace."""
    if not text:
        return ""

    replacements = {
        "â€™": "’",
        "â€˜": "‘",
        "â€œ": "“",
        "â€\x9d": "”",
        "â€“": "–",
        "â€”": "—",
        "â€¦": "…",
        "â€¢": "•",
        "Â©": "©",
        "Â®": "®",
        "Â ": " ",
    }

    for broken, corrected in replacements.items():
        text = text.replace(broken, corrected)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]

    cleaned_lines = []
    previous_blank = False

    for line in lines:
        if not line:
            if cleaned_lines and not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
        else:
            cleaned_lines.append(line)
            previous_blank = False

    return "\n".join(cleaned_lines).strip()
