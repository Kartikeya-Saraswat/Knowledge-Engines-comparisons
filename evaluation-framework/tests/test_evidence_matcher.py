
import json
from pathlib import Path

import pytest

from src.evaluation.evidence_matcher import (
    annotate_retrieval_results, match_pdf_anchor, match_okf_concept,
    match_result_to_evidence_units,
)


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


def test_partial_evidence_from_chunk_boundary_does_not_receive_full_credit():
    result = make_pdf_result(
        "On the basis of exhaustibility renewable and non-renewable."
    )
    assert not match_pdf_anchor(result, ANCHOR)


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


def audited_questions():
    path = Path(__file__).resolve().parents[1] / "benchmark/questions.json"
    return {q["id"]: q for q in json.loads(path.read_text(encoding="utf-8"))}


def test_complete_atomic_fact_does_not_need_other_facts_in_context_quote():
    units = audited_questions()["JESS101-Q01"]["evidence_units"]
    result = make_pdf_result("On the basis of exhaustibility - renewable and non-renewable.")
    assert {m["evidence_unit_id"] for m in match_result_to_evidence_units(result, units)} == {"E2"}


def test_one_waterway_does_not_earn_credit_for_the_other_four():
    units = audited_questions()["JESS107-Q04"]["evidence_units"]
    result = make_pdf_result(units[0]["pdf_anchors"][0]["quote"], document="jess107.pdf", pages=[7])
    assert {m["evidence_unit_id"] for m in match_result_to_evidence_units(result, units)} == {"E1", "E6"}


def test_wrong_waterway_length_is_not_a_match():
    unit = audited_questions()["JESS107-Q04"]["evidence_units"][5]
    source = unit["pdf_anchors"][0]["quote"]
    result = make_pdf_result(source.replace("1620", "999"), document="jess107.pdf", pages=[7])
    assert match_result_to_evidence_units(result, [unit]) == []


def test_waterway_number_one_is_not_number_ten():
    unit = audited_questions()["JESS107-Q04"]["evidence_units"][0]
    source = unit["pdf_anchors"][0]["quote"]
    result = make_pdf_result(source.replace("No.1", "No.10"), document="jess107.pdf", pages=[7])
    assert not match_result_to_evidence_units(result, [unit])


@pytest.mark.parametrize("replacement", ["42 : 1", "4 : 21", "4 : 2 : 10", "4.21"])
def test_changed_numeric_ratios_are_not_formatting_differences(replacement):
    unit = audited_questions()["JESS106-Q04"]["evidence_units"][1]
    source = unit["pdf_anchors"][0]["quote"]
    result = make_pdf_result(source.replace("4 : 2 : 1", replacement), document="jess106.pdf", pages=[5])
    assert not match_result_to_evidence_units(result, [unit])


def test_ratio_spacing_is_formatting_only():
    unit = audited_questions()["JESS106-Q04"]["evidence_units"][1]
    source = unit["pdf_anchors"][0]["quote"]
    result = make_pdf_result(source.replace("4 : 2 : 1", "4:2:1"), document="jess106.pdf", pages=[5])
    assert match_result_to_evidence_units(result, [unit])


@pytest.mark.parametrize("quote,text", [
    ("The measured value is 1.5.", "The measured value is 15."),
    ("The measured value is 10%.", "The measured value is 10."),
])
def test_numeric_meaning_is_not_removed_with_formatting_punctuation(quote, text):
    anchor = {**ANCHOR, "quote": quote}
    assert not match_pdf_anchor(make_pdf_result(text), anchor)


def test_thousands_separator_is_formatting_only():
    anchor = {**ANCHOR, "quote": "The length is 1,620 km."}
    assert match_pdf_anchor(make_pdf_result("The length is 1620 km."), anchor)


@pytest.mark.parametrize("text", [
    "On the basis of exhaustibility renewable.",
    "Resources renewable non-renewable exhaustibility.",
    "On the basis of exhaustibility these are not renewable and non-renewable.",
    "On the basis of exhaustibility non-renewable and renewable.",
])
def test_missing_changed_or_reordered_words_do_not_pass_as_complete_evidence(text):
    unit = audited_questions()["JESS101-Q01"]["evidence_units"][1]
    assert match_result_to_evidence_units(make_pdf_result(text), [unit]) == []


def test_required_months_cannot_be_replaced_by_generic_seasons():
    units = audited_questions()["JESS104-Q04"]["evidence_units"]
    result = make_pdf_result("Rabi crops are sown in winter and harvested in summer.",
                             document="jess104.pdf", pages=[3])
    assert match_result_to_evidence_units(result, units) == []
    full = make_pdf_result("Rabi crops are sown in winter from October to December and harvested in summer from April to June.",
                           document="jess104.pdf", pages=[3])
    assert {m["evidence_unit_id"] for m in match_result_to_evidence_units(full, units)} == {"E2", "E3"}


def test_nearly_complete_quote_still_needs_its_final_words():
    assert not match_pdf_anchor(make_pdf_result(ANCHOR["quote"][:-10]), ANCHOR)


def test_partial_records_are_not_concatenated_into_one_complete_match():
    unit = audited_questions()["JESS101-Q01"]["evidence_units"][1]
    records = [make_pdf_result("On the basis of exhaustibility renewable"),
               make_pdf_result("and non-renewable.")]
    assert all(not item["matches"] for item in annotate_retrieval_results(records, [unit]))


def test_unicode_case_wrapping_and_punctuation_are_formatting_only():
    unit = audited_questions()["JESS101-Q01"]["evidence_units"][1]
    result = make_pdf_result("ON THE BASIS OF EXHAUSTIBILITY: renewable and non-\nrenewable.")
    assert match_result_to_evidence_units(result, [unit])


@pytest.mark.parametrize("document", [r"C:\books\JESS101.PDF", "/books/JESS101.PDF"])
def test_document_paths_are_normalized_across_platforms(document):
    assert match_pdf_anchor(make_pdf_result(ANCHOR["quote"], document=document), ANCHOR)


def test_missing_or_malformed_metadata_cannot_supply_provenance():
    for metadata in [None, {}, "invalid"]:
        result = {"text": ANCHOR["quote"], "metadata": metadata}
        assert not match_pdf_anchor(result, ANCHOR)


@pytest.mark.parametrize("pages", [None, [], [True], [0], [-1], ["1"], "not-json", "[true]", "[0]", "[\"1\"]"])
def test_missing_or_invalid_page_provenance_fails_closed(pages):
    result = {"text": ANCHOR["quote"], "document_name": "jess101.pdf", "page_numbers": pages}
    assert not match_pdf_anchor(result, ANCHOR)


@pytest.mark.parametrize("pages", [[1], (1,), "[1]"])
def test_native_and_serialized_page_provenance_are_supported(pages):
    result = {"text": ANCHOR["quote"], "document_name": "jess101.pdf", "page_numbers": pages}
    assert match_pdf_anchor(result, ANCHOR)


def test_multi_page_anchor_requires_all_cited_pages():
    anchor = {**ANCHOR, "pages": [1, 2]}
    assert not match_pdf_anchor(make_pdf_result(ANCHOR["quote"], pages=[1]), anchor)
    assert match_pdf_anchor(make_pdf_result(ANCHOR["quote"], pages=[1, 2]), anchor)


@pytest.mark.parametrize("text", [None, "", "  ", 42])
def test_missing_text_fails_closed(text):
    assert not match_pdf_anchor(make_pdf_result(text), ANCHOR)


@pytest.mark.parametrize("required", [None, [], [""], ["!!!"], ["not in the source"], [42]])
def test_invalid_required_excerpts_fail_closed(required):
    anchor = {**ANCHOR, "required_quotes": required}
    assert not match_pdf_anchor(make_pdf_result(ANCHOR["quote"]), anchor)


def test_alternative_anchor_must_pass_provenance_and_content():
    unit = {"id": "E1", "relevance": 3, "pdf_anchor_mode": "any",
            "pdf_anchors": [{**ANCHOR, "pages": [8]}, ANCHOR]}
    assert match_result_to_evidence_units(make_pdf_result(ANCHOR["quote"]), [unit])
    assert not match_result_to_evidence_units(make_pdf_result("unrelated"), [unit])


def test_joint_anchor_modes_are_not_silently_treated_as_alternatives():
    unit = {"id": "E1", "pdf_anchor_mode": "all", "pdf_anchors": [ANCHOR]}
    with pytest.raises(ValueError, match="Joint PDF anchors"):
        match_result_to_evidence_units(make_pdf_result(ANCHOR["quote"]), [unit])


def test_okf_matching_requires_full_audited_text_not_just_the_id():
    unit = audited_questions()["JESS106-Q04"]["evidence_units"][1]
    support = unit["okf_support"][0]
    result = {"record_id": support["concept_id"], "source_type": "okf", "text": "Unrelated text."}
    assert not match_result_to_evidence_units(result, [unit])
    result["text"] = "\n".join(support["quotes"])
    assert match_result_to_evidence_units(result, [unit])[0]["matched_by"] == ["okf"]
    result["text"] = result["text"].replace("4:2:1", "1:1:1")
    assert not match_result_to_evidence_units(result, [unit])


def test_partial_okf_support_never_receives_full_credit():
    unit = audited_questions()["JESS104-Q04"]["evidence_units"][1]
    partial = unit["okf_support"][0]
    result = {"record_id": partial["concept_id"], "source_type": "okf",
              "text": "\n".join(partial["quotes"])}
    assert not match_result_to_evidence_units(result, [unit])
    # Even a malformed benchmark that promotes the ID must not bypass the label.
    bad_unit = {**unit, "okf_concept_ids": [partial["concept_id"]]}
    assert not match_result_to_evidence_units(result, [bad_unit])


def test_hybrid_records_are_scored_using_their_underlying_source():
    unit = audited_questions()["JESS101-Q01"]["evidence_units"][1]
    result = make_pdf_result("On the basis of exhaustibility renewable and non-renewable.")
    result.update(source_type="hybrid", record_id="pdf-hash")
    annotated = annotate_retrieval_results([result], [unit])
    assert annotated[0]["matches"][0]["matched_by"] == ["pdf"]
    support = unit["okf_support"][0]
    result = {"record_id": support["concept_id"], "source_type": "hybrid",
              "metadata": {"source_type": "okf", "concept_id": support["concept_id"]},
              "text": "\n".join(support["quotes"])}
    assert annotate_retrieval_results([result], [unit])[0]["matches"][0]["matched_by"] == ["okf"]


def test_every_audited_pdf_anchor_matches_its_complete_source_quote():
    for question in audited_questions().values():
        for unit in question["evidence_units"]:
            for anchor in unit["pdf_anchors"]:
                result = make_pdf_result(anchor["quote"], document=anchor["document"], pages=anchor["pages"])
                assert match_pdf_anchor(result, anchor), (question["id"], unit["id"])
