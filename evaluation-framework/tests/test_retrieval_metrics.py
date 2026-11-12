
import pytest

from src.metrics.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    evaluate_retrieval,
)


@pytest.fixture
def evidence_units():
    return [
        {"id": "E1", "relevance": 3},
        {"id": "E2", "relevance": 2},
    ]


@pytest.fixture
def annotated_results():
    return [
        {
            "rank": 1,
            "matches": [
                {"evidence_unit_id": "E1", "relevance": 3}
            ],
        },
        {
            "rank": 2,
            "matches": [
                {"evidence_unit_id": "E1", "relevance": 3}
            ],
        },
        {
            "rank": 3,
            "matches": [
                {"evidence_unit_id": "E2", "relevance": 2}
            ],
        },
        {"rank": 4, "matches": []},
    ]


def test_precision_at_k_counts_relevant_chunks(annotated_results):
    assert precision_at_k(annotated_results, 3) == 1.0


def test_recall_at_k_counts_unique_evidence(
    annotated_results,
):
    assert recall_at_k(annotated_results, 2, 3) == 1.0


def test_duplicate_evidence_does_not_inflate_recall(
    annotated_results,
):
    assert recall_at_k(annotated_results[:2], 2, 2) == 0.5


def test_mrr_uses_first_relevant_rank(annotated_results):
    assert reciprocal_rank(annotated_results) == 1.0


def test_ndcg_is_excluded(annotated_results, evidence_units):
    score = ndcg_at_k(annotated_results, evidence_units, 3)
    assert score is None


def test_multiple_evidence_units_in_one_chunk():
    results = [
        {
            "rank": 1,
            "matches": [
                {"evidence_unit_id": "E1", "relevance": 3},
                {"evidence_unit_id": "E2", "relevance": 2},
            ],
        }
    ]

    # One retrieved chunk, even though it matches two evidence units.
    assert precision_at_k(results, 1) == 1.0
    assert recall_at_k(results, 2, 1) == 1.0
    assert reciprocal_rank(results) == 1.0


def test_empty_retrieval(evidence_units):
    assert precision_at_k([], 5) == 0.0
    assert recall_at_k([], 2, 5) == 0.0
    assert reciprocal_rank([]) == 0.0
    assert ndcg_at_k([], evidence_units, 5) is None


def test_unanswerable_question_with_no_evidence():
    assert recall_at_k([], 0, 5) is None
    assert ndcg_at_k([], [], 5) is None


def test_invalid_k(annotated_results):
    with pytest.raises(ValueError):
        precision_at_k(annotated_results, 0)


def test_evaluate_retrieval_returns_expected_metrics(
    annotated_results,
    evidence_units,
):
    result = evaluate_retrieval(
        annotated_results,
        evidence_units,
        k_values=(1, 3),
    )

    assert result["mrr"] == 1.0
    assert result["precision"][3] == 1.0
    assert result["recall"][3] == 1.0
    assert result["map"] is None
    assert result["complete_success"][3] == 1.0
    assert result["duplicate_rate"][3] == pytest.approx(1 / 3)


def test_complete_evidence_is_independent_of_record_packing(evidence_units):
    packed = [{"matches": [{"evidence_unit_id": "E1", "relevance": 3}, {"evidence_unit_id": "E2", "relevance": 2}]}]
    split = [{"matches": [hit]} for hit in packed[0]["matches"]]
    for results in (packed, split):
        metrics = evaluate_retrieval(results, evidence_units, (2,))
        assert metrics["recall"][2] == metrics["complete_success"][2] == 1.0
        assert metrics["ndcg"][2] is None


def test_mrr_respects_collected_depth_and_excludes_unanswerable():
    records = [{'matches': []}, {'matches': [{'evidence_unit_id': 'E1', 'relevance': 3}]}]
    assert evaluate_retrieval(records, [{'id': 'E1', 'relevance': 3}], (1,))['mrr'] == 0
    assert evaluate_retrieval([], [], (1,))['mrr'] is None
