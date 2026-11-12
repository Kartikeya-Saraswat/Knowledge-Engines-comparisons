
from src.evaluation.evidence_matcher import match_pdf_anchor, match_okf_concept


ANCHOR = {
    "document": "jess101.pdf",
    "pages": [1],
    "quote": (
        "These resources can be classified in the following ways "
        "on the basis of origin biotic and abiotic "
        "on the basis of exhaustibility renewable and non-renewable"
    ),
}


def make_pdf_result(text, document="jess101.pdf", pages=None):
    return {
        "text": text,
        "metadata": {
            "source_type": "pdf",
            "document_name": document,
            "page_numbers": [1] if pages is None else pages,
        },
    }


def test_matching_document_page_and_evidence():
    result = make_pdf_result(
        "These resources can be classified in the following ways "
        "on the basis of origin biotic and abiotic "
        "on the basis of exhaustibility renewable and non renewable"
    )
    assert match_pdf_anchor(result, ANCHOR)


def test_same_document_and_page_with_unrelated_text():
    result = make_pdf_result(
        "Land resources are used for agriculture, forests and settlements."
    )
    assert not match_pdf_anchor(result, ANCHOR)


def test_matching_evidence_with_punctuation_differences():
    result = make_pdf_result(
        "These resources can be classified in the following ways: "
        "on the basis of origin - biotic and abiotic; "
        "on the basis of exhaustibility - renewable and non-renewable."
    )
    assert match_pdf_anchor(result, ANCHOR)


def test_matching_partial_evidence_from_chunk_boundary():
    result = make_pdf_result(
        "On the basis of exhaustibility renewable and non-renewable."
    )
    assert match_pdf_anchor(result, ANCHOR)


def test_correct_text_wrong_page():
    result = make_pdf_result(ANCHOR["quote"], pages=[5])
    assert not match_pdf_anchor(result, ANCHOR)


def test_correct_text_wrong_document():
    result = make_pdf_result(
        ANCHOR["quote"],
        document="jess102.pdf",
    )
    assert not match_pdf_anchor(result, ANCHOR)


def test_okf_requires_exact_concept_id():
    result = {
        "record_id": "KC-JESS101-001",
        "metadata": {"source_type": "okf"},
    }

    assert match_okf_concept(result, "KC-JESS101-001")
    assert not match_okf_concept(result, "KC-JESS101-002")
