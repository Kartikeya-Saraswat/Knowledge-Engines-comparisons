import pytest

from src.fusion.ranker import (
    reciprocal_rank_fusion,
    DEFAULT_RRF_K,
)
from src.retrieval.vector_store import RetrievedRecord


def make_result(record_id, source_type, text):
    return RetrievedRecord(
        record_id=record_id,
        text=text,
        metadata={"source_type": source_type},
        distance=0.1,
    )


def test_fuses_pdf_and_okf_results():
    pdf = [
        make_result("A", "pdf", "PDF A"),
        make_result("B", "pdf", "PDF B"),
    ]

    okf = [
        make_result("C", "okf", "OKF C"),
        make_result("A", "okf", "OKF A"),
    ]

    results = reciprocal_rank_fusion(pdf, okf)

    assert len(results) == 3
    assert results[0].record_id == "A"
    assert results[0].source_ranks == {"pdf": 1, "okf": 2}


def test_rrf_score_is_calculated_correctly():
    pdf = [make_result("A", "pdf", "Content A")]
    okf = [make_result("A", "okf", "Content A")]

    results = reciprocal_rank_fusion(pdf, okf)

    expected_score = (
        1 / (DEFAULT_RRF_K + 1)
        + 1 / (DEFAULT_RRF_K + 1)
    )

    assert results[0].rrf_score == pytest.approx(expected_score)


def test_preserves_independent_source_rankings():
    pdf = [
        make_result("A", "pdf", "A"),
        make_result("B", "pdf", "B"),
    ]

    okf = [
        make_result("B", "okf", "B"),
        make_result("A", "okf", "A"),
    ]

    results = reciprocal_rank_fusion(pdf, okf)

    result_b = next(r for r in results if r.record_id == "B")

    assert result_b.source_ranks == {"pdf": 2, "okf": 1}


def test_empty_results():
    assert reciprocal_rank_fusion([], []) == []


def test_pdf_only_results():
    pdf = [
        make_result("A", "pdf", "A"),
        make_result("B", "pdf", "B"),
    ]

    results = reciprocal_rank_fusion(pdf, [])

    assert len(results) == 2
    assert results[0].record_id == "A"
    assert results[0].source_ranks == {"pdf": 1}


def test_okf_only_results():
    okf = [
        make_result("A", "okf", "A"),
    ]

    results = reciprocal_rank_fusion([], okf)

    assert len(results) == 1
    assert results[0].source_ranks == {"okf": 1}


def test_duplicate_record_in_same_source_counts_once():
    pdf = [
        make_result("A", "pdf", "A"),
        make_result("A", "pdf", "A duplicate"),
    ]

    results = reciprocal_rank_fusion(pdf, [])

    assert len(results) == 1
    assert results[0].source_ranks == {"pdf": 1}
    assert results[0].rrf_score == pytest.approx(
        1 / (DEFAULT_RRF_K + 1)
    )


def test_invalid_rrf_constant():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], [], rrf_k=0)