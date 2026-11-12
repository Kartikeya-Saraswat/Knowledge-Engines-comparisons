
import math


def _unique_evidence_hits(annotated_results):
    """
    Keep only the first occurrence of each evidence unit.

    Repeated chunks matching the same evidence unit receive
    no additional relevance credit.
    """
    seen = set()
    hits = []

    for result in annotated_results:
        result_hits = []

        for match in result.get("matches", []):
            evidence_id = match["evidence_unit_id"]

            if evidence_id in seen:
                continue

            seen.add(evidence_id)
            result_hits.append(match)

        hits.append(result_hits)

    return hits


def precision_at_k(annotated_results, k):
    if k < 1:
        raise ValueError("k must be positive.")

    relevant_count = sum(bool(result.get("matches")) for result in annotated_results[:k])

    return relevant_count / k


def recall_at_k(annotated_results, total_relevant, k):
    if k < 1:
        raise ValueError("k must be positive.")

    if total_relevant <= 0:
        return None

    hits = _unique_evidence_hits(annotated_results[:k])
    retrieved_relevant = sum(
        len(result) for result in hits
    )

    return retrieved_relevant / total_relevant


def reciprocal_rank(annotated_results):
    hits = _unique_evidence_hits(annotated_results)

    for rank, result_hits in enumerate(hits, start=1):
        if result_hits:
            return 1 / rank

    return 0.0


def dcg_at_k(annotated_results, k):
    if k < 1:
        raise ValueError("k must be positive.")

    hits = _unique_evidence_hits(annotated_results[:k])
    score = 0.0

    for rank, result_hits in enumerate(hits, start=1):
        # A retrieved chunk receives the highest grade
        # among the distinct evidence units it matches.
        relevance = max(
            (match["relevance"] for match in result_hits),
            default=0,
        )

        gain = (2 ** relevance) - 1
        score += gain / math.log2(rank + 1)

    return score

def ndcg_at_k(annotated_results, evidence_units, k):
    if k < 1:
        raise ValueError("k must be positive.")

    ideal_relevances = sorted(
        (unit["relevance"] for unit in evidence_units),
        reverse=True,
    )[:k]

    ideal_dcg = sum(
        ((2 ** relevance) - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevances, start=1)
    )

    if ideal_dcg == 0:
        return None

    # Excluded: an evidence-unit ideal is not comparable to record-level gain
    # when records can cover several facts. Do not publish a misleading nDCG.
    return None


def evaluate_retrieval(
    annotated_results,
    evidence_units,
    k_values=(1, 3, 5, 8),
):
    """
    Calculate retrieval metrics for one question and one arm.
    """
    total_relevant = len(evidence_units)
    if not k_values or any(type(k) is not int or k < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers")

    metrics = {
        "mrr": reciprocal_rank(annotated_results[:max(k_values)]) if total_relevant else None,
        "map": None,
        "precision": {},
        "recall": {},
        "ndcg": {},
        "complete_success": {},
        "duplicate_rate": {},
    }

    for k in k_values:
        metrics["precision"][k] = precision_at_k(
            annotated_results, k
        )

        metrics["recall"][k] = recall_at_k(
            annotated_results, total_relevant, k
        )

        metrics["ndcg"][k] = ndcg_at_k(
            annotated_results, evidence_units, k
        )
        metrics["complete_success"][k] = (
            float(metrics["recall"][k] == 1.0) if total_relevant else None
        )
        hits = [m["evidence_unit_id"] for r in annotated_results[:k] for m in r.get("matches", [])]
        metrics["duplicate_rate"][k] = (len(hits) - len(set(hits))) / len(hits) if hits else 0.0

    return metrics
