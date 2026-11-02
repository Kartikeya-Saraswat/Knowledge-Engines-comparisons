
from src.ingestion.page_mapping import (
    detect_printed_page_label,
    roman_to_int,
)


def test_detects_arabic_page_label():
    result = detect_printed_page_label(
        "GEOGRAPHY\nResources and Development\n\n49"
    )

    assert result["page_label"] == "49"
    assert result["printed_page_number"] == 49
    assert result["page_label_type"] == "arabic"


def test_detects_roman_page_label():
    result = detect_printed_page_label(
        "PREFACE\n\niv"
    )

    assert result["page_label"] == "iv"
    assert result["printed_page_number"] == 4
    assert result["page_label_type"] == "roman"


def test_missing_page_label_returns_none():
    result = detect_printed_page_label(
        "CONTENTS\nChapter One\nChapter Two"
    )

    assert result["page_label"] is None
    assert result["printed_page_number"] is None


def test_rejects_malformed_roman_numeral():
    assert roman_to_int("iiv") is None
    assert roman_to_int("vx") is None
    assert roman_to_int("iv") == 4

def test_ignores_roman_numeral_in_textbook_title():
    result = detect_printed_page_label(
        "India\nContemporary\nII\nTextbook in Geography\n"
        "for Class X\nSocial Science\nReprint 2026-27"
    )

    assert result["page_label"] is None
    assert result["printed_page_number"] is None


def test_detects_roman_page_number_at_bottom():
    result = detect_printed_page_label(
        "Preface\nAcknowledgements\nDirector\n"
        "National Council of Educational Research and Training\niv"
    )

    assert result["page_label"] == "iv"
    assert result["printed_page_number"] == 4
    assert result["page_label_type"] == "roman"

def test_ignores_standalone_number_in_middle_of_page():
    result = detect_printed_page_label(
        "GEOGRAPHY\n"
        "The chapter discusses resources.\n"
        "49\n"
        "This number appears in the middle of the extracted text.\n"
        "More content follows."
    )

    assert result["page_label"] is None


def test_ignores_year_at_bottom_of_page():
    result = detect_printed_page_label(
        "GEOGRAPHY\n"
        "Resources and Development\n"
        "Reprint 2026-27"
    )

    assert result["page_label"] is None


def test_ignores_number_followed_by_ordinary_content():
    result = detect_printed_page_label(
        "GEOGRAPHY\n"
        "Resources and Development\n"
        "49\n"
        "Chapter content"
    )

    assert result["page_label"] is None