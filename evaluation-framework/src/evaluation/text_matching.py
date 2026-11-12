"""Conservative excerpt matching shared by annotation validation and scoring."""

import re
import unicodedata


def normalize_excerpt(text):
    """Ignore formatting; retain letters, digits, and meaningful numeric separators.

    This tolerates wrapped lines, split hyphens and PDF punctuation differences.
    It does not drop words, numbers, negations, or repeated extraction tokens.
    """
    text = unicodedata.normalize("NFKC", text).casefold().replace("\u00ad", "")
    # A ratio of 4:2:1 must never normalize to the different ratio 42:1 or
    # decimal 4.21. Thousands commas are formatting; decimal/fraction/ratio
    # separators and percent units carry meaning.
    numeric_separators = {":": "numericcolon", ".": "numericdecimal", "/": "numericfraction"}
    text = re.sub(r"(?<=\d)\s*([:./])\s*(?=\d)",
                  lambda match: numeric_separators[match.group(1)], text)
    text = text.replace("%", "percent")
    return re.sub(r"[\W_]+", "", text)


def contains_required_quotes(text, quotes):
    """Require every complete excerpt in order within one retrieved record.

    No partial substring of a required excerpt, keyword overlap, or fuzzy
    similarity earns credit. Multiple excerpts may omit irrelevant source context.
    """
    if not isinstance(text, str) or not isinstance(quotes, list) or not quotes:
        return False
    normalized = normalize_excerpt(text)
    if not normalized:
        return False
    offset = 0
    for quote in quotes:
        if not isinstance(quote, str):
            return False
        required = normalize_excerpt(quote)
        if not required:
            return False
        position = normalized.find(required, offset)
        # Do not match an excerpt ending in No.1 against No.10, or a required
        # numeric value as a prefix/suffix of a different value.
        while position >= 0:
            end = position + len(required)
            wrong_start = (required[0].isdigit() and position > 0
                           and normalized[position - 1].isdigit())
            wrong_end = (required[-1].isdigit() and end < len(normalized)
                         and normalized[end].isdigit())
            if not wrong_start and not wrong_end:
                break
            position = normalized.find(required, position + 1)
        if position < 0:
            return False
        offset = position + len(required)
    return True
