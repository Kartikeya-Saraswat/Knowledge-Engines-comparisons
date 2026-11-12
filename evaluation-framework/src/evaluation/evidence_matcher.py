
import json

from src.evaluation.text_matching import contains_required_quotes


def parse_list(value):
    """Parse list values stored as native lists or JSON strings."""
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return list(value)

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []

    return []


def normalize_document_name(name):
    """Normalize document names for reliable comparison."""
    return str(name).replace("\\", "/").rsplit("/", 1)[-1].casefold()


def valid_pages(value):
    pages = parse_list(value)
    if not pages or any(type(page) is not int or page < 1 for page in pages):
        return set()
    return set(pages)


def match_pdf_anchor(result, anchor):
    """Require source provenance and complete source excerpts in the chunk."""
    metadata = result.get("metadata") or {}
    if not isinstance(metadata, dict):
        return False

    document = (
        result.get("document_name")
        or metadata.get("document_name")
    )

    pages = result.get("page_numbers")
    if pages is None:
        pages = metadata.get("page_numbers")
    retrieved_pages = valid_pages(pages)
    anchor_pages = valid_pages(anchor.get("pages"))

    if (not isinstance(document, str) or not document.strip()
            or not isinstance(anchor.get("document"), str) or not anchor["document"].strip()
            or not retrieved_pages or not anchor_pages):
        return False

    if normalize_document_name(document) != normalize_document_name(
        anchor["document"]
    ):
        return False

    if not anchor_pages.issubset(retrieved_pages):
        return False
    quote = anchor.get("quote")
    if not isinstance(quote, str) or not quote.strip():
        return False
    required = anchor.get("required_quotes", [quote])
    # Fail closed if matching requirements have drifted from the audited source.
    return (contains_required_quotes(quote, required)
            and contains_required_quotes(result.get("text"), required))


def match_okf_concept(result, concept_id):
    """Match a retrieved OKF concept by its exact concept ID."""
    metadata = result.get("metadata") or {}
    if not isinstance(metadata, dict):
        return False

    retrieved_id = (
        metadata.get("concept_id")
        or result.get("record_id")
    )

    return retrieved_id == concept_id


def match_result_to_evidence_units(result, evidence_units):
    """
    Match one retrieval result against benchmark evidence units.

    Returns each matched evidence unit once, preserving its
    benchmark relevance grade.
    """
    metadata = result.get("metadata") or {}
    if not isinstance(metadata, dict):
        return []

    source_type = (
        metadata.get("source_type")
        or result.get("source_type")
    )

    matches = []

    for unit in evidence_units:
        if unit.get("pdf_anchor_mode", "any") != "any":
            raise ValueError("Joint PDF anchors are unsupported; split them into atomic evidence units")
        matched_by = []

        if source_type in ("pdf", "hybrid"):
            for anchor in unit.get("pdf_anchors", []):
                if match_pdf_anchor(result, anchor):
                    matched_by.append("pdf")
                    break

        if source_type in ("okf", "hybrid"):
            for concept_id in unit.get("okf_concept_ids", []):
                full_support = next((
                    support for support in unit.get("okf_support", [])
                    if support.get("concept_id") == concept_id and support.get("support") == "full"
                ), None)
                if (match_okf_concept(result, concept_id) and full_support
                        and contains_required_quotes(result.get("text"), full_support.get("quotes"))):
                    matched_by.append("okf")
                    break

        if matched_by:
            matches.append({
                "evidence_unit_id": unit["id"],
                "relevance": unit["relevance"],
                "matched_by": matched_by,
            })

    return matches


def annotate_retrieval_results(results, evidence_units):
    """Attach benchmark evidence matches to ranked results."""
    annotated = []

    for rank, result in enumerate(results, start=1):
        annotated.append({
            "rank": rank,
            "record_id": result.get("record_id"),
            "matches": match_result_to_evidence_units(
                result,
                evidence_units,
            ),
        })

    return annotated
