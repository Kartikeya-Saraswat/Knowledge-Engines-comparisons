from dataclasses import dataclass

from src.retrieval.vector_store import RetrievedRecord


DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedRecord:
    record_id: str
    text: str
    metadata: dict
    rrf_score: float
    source_ranks: dict[str, int]


def reciprocal_rank_fusion(
    pdf_results: list[RetrievedRecord],
    okf_results: list[RetrievedRecord],
    rrf_k: int = DEFAULT_RRF_K,
) -> list[FusedRecord]:
    """Combine PDF and OKF rankings using Reciprocal Rank Fusion."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")

    fused = {}

    sources = {
        "pdf": pdf_results,
        "okf": okf_results,
    }

    for source_name, results in sources.items():
        seen_in_source = set()

        for rank, result in enumerate(results, start=1):
            # A record should contribute only once per source.
            if result.record_id in seen_in_source:
                continue

            seen_in_source.add(result.record_id)

            if result.record_id not in fused:
                fused[result.record_id] = {
                    "record": result,
                    "score": 0.0,
                    "source_ranks": {},
                }

            fused[result.record_id]["score"] += (
                1.0 / (rrf_k + rank)
            )

            fused[result.record_id]["source_ranks"][
                source_name
            ] = rank

    ranked_records = [
        FusedRecord(
            record_id=item["record"].record_id,
            text=item["record"].text,
            metadata=item["record"].metadata,
            rrf_score=item["score"],
            source_ranks=item["source_ranks"],
        )
        for item in fused.values()
    ]

    return sorted(
        ranked_records,
        key=lambda record: record.rrf_score,
        reverse=True,
    )