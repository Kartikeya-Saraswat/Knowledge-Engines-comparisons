
import json
from pathlib import Path


def parse_list(value):
    """Parse list values stored as native lists or JSON strings."""
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []

    return []


def normalize_document_name(name):
    """Normalize document names for reliable comparison."""
    return Path(str(name)).name.casefold()


def match_pdf_anchor(result, anchor):
    """Match a retrieved PDF chunk against a benchmark PDF anchor."""
    metadata = result.get("metadata") or {}

    document = (
        result.get("document_name")
        or metadata.get("document_name")
    )

    pages = (
        result.get("page_numbers")
        or parse_list(metadata.get("page_numbers"))
        or parse_list(metadata.get("source_pages"))
    )

    if not document or not pages:
        return False

    if normalize_document_name(document) != normalize_document_name(
        anchor["document"]
    ):
        return False

    anchor_pages = set(anchor.get("pages", []))
    retrieved_pages = set(pages)

    return bool(anchor_pages.intersection(retrieved_pages))


def match_okf_concept(result, concept_id):
    """Match a retrieved OKF concept by its exact concept ID."""
    metadata = result.get("metadata") or {}

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

    source_type = (
        metadata.get("source_type")
        or result.get("source_type")
    )

    matches = []

    for unit in evidence_units:
        matched_by = []

        if source_type in ("pdf", "hybrid"):
            for anchor in unit.get("pdf_anchors", []):
                if match_pdf_anchor(result, anchor):
                    matched_by.append("pdf")
                    break

        if source_type in ("okf", "hybrid"):
            for concept_id in unit.get("okf_concept_ids", []):
                if match_okf_concept(result, concept_id):
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
