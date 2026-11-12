
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import load_config, resolve_path
from src.evaluation.evidence_matcher import annotate_retrieval_results
from src.metrics.retrieval_metrics import evaluate_retrieval


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def mean(values):
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def aggregate_metrics(records, k_values):
    """Macro-average metrics across questions."""
    summary = {
        "question_count": len(records),
        "mrr": mean([r["metrics"]["mrr"] for r in records]),
        "map": mean([r["metrics"]["map"] for r in records]),
        "precision": {},
        "recall": {},
        "ndcg": {},
    }

    for k in k_values:
        summary["precision"][str(k)] = mean([
            r["metrics"]["precision"][k] for r in records
        ])
        summary["recall"][str(k)] = mean([
            r["metrics"]["recall"][k] for r in records
        ])
        summary["ndcg"][str(k)] = mean([
            r["metrics"]["ndcg"][k] for r in records
        ])

    return summary


def run_evaluation(benchmark, raw_results, k_values):
    questions = {q["id"]: q for q in benchmark}
    detailed_records = []
    seen_pairs = set()

    for retrieval in raw_results["retrieval_records"]:
        question_id = retrieval["question_id"]
        arm = retrieval["arm"]

        if question_id not in questions:
            raise ValueError(
                f"Retrieval references unknown question: {question_id}"
            )

        pair = (question_id, arm)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate retrieval record: {pair}")
        seen_pairs.add(pair)

        question = questions[question_id]
        evidence_units = question.get("evidence_units", [])

        annotated = annotate_retrieval_results(
            retrieval["results"],
            evidence_units,
        )

        metrics = evaluate_retrieval(
            annotated,
            evidence_units,
            k_values,
        )

        detailed_records.append({
            "question_id": question_id,
            "question_type": question["type"],
            "arm": arm,
            "answerable": question.get("answerable", True),
            "evidence_unit_count": len(evidence_units),
            "retrieved_count": len(retrieval["results"]),
            "annotated_results": annotated,
            "metrics": metrics,
        })

    expected_pairs = {
        (question_id, arm)
        for question_id in questions
        for arm in raw_results["arms"]
    }

    missing_pairs = expected_pairs - seen_pairs
    if missing_pairs:
        raise ValueError(
            f"Missing retrieval records: {sorted(missing_pairs)}"
        )

    return detailed_records


def build_summaries(records, k_values):
    by_arm = defaultdict(list)
    by_type = defaultdict(list)

    for record in records:
        by_arm[record["arm"]].append(record)
        by_type[(record["arm"], record["question_type"])].append(record)

    arm_summary = {
        arm: aggregate_metrics(items, k_values)
        for arm, items in sorted(by_arm.items())
    }

    type_summary = {
        arm: {
            question_type: aggregate_metrics(items, k_values)
            for (record_arm, question_type), items in sorted(
                by_type.items()
            )
            if record_arm == arm
        }
        for arm in sorted(by_arm)
    }

    return arm_summary, type_summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate knowledge retrieval results."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    benchmark_path = resolve_path(
        config["benchmark"]["file"], config
    )

    if args.input:
        raw_path = resolve_path(args.input, config)
    else:
        results_directory = resolve_path(
            config["outputs"]["results_directory"], config
        )
        candidates = sorted(
            results_directory.glob("raw_retrieval_*.json")
        )
        if not candidates:
            raise FileNotFoundError(
                "No raw retrieval result files found."
            )
        raw_path = candidates[-1]

    benchmark = load_json(benchmark_path)
    raw_results = load_json(raw_path)
    k_values = config["evaluation"]["retrieval_top_k"]

    print(f"Benchmark questions: {len(benchmark)}")
    print(f"Retrieval records: {len(raw_results['retrieval_records'])}")
    print(f"Input: {raw_path}")

    records = run_evaluation(
        benchmark,
        raw_results,
        k_values,
    )

    arm_summary, type_summary = build_summaries(
        records,
        k_values,
    )

    output_directory = resolve_path(
        config["outputs"]["results_directory"], config
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    detailed_path = output_directory / (
        f"detailed_evaluation_{timestamp}.json"
    )
    summary_path = output_directory / (
        f"summary_evaluation_{timestamp}.json"
    )

    metadata = {
        "experiment": config["project"]["name"],
        "created_at": timestamp,
        "benchmark": str(benchmark_path),
        "raw_results": str(raw_path),
        "question_count": len(benchmark),
        "retrieval_record_count": len(records),
        "arms": raw_results["arms"],
        "k_values": k_values,
    }

    detailed_output = {
        "metadata": metadata,
        "records": records,
    }

    summary_output = {
        "metadata": metadata,
        "by_arm": arm_summary,
        "by_question_type": type_summary,
    }

    detailed_path.write_text(
        json.dumps(detailed_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n--- Overall Results ---")
    for arm, metrics in arm_summary.items():
        print(f"\n{arm}")
        print(f"  MRR: {metrics['mrr']:.4f}")
        if metrics["map"] is not None:
            print(f"  MAP: {metrics['map']:.4f}")
        else:
            print("  MAP: Not calculated (no chunk-level relevance labels)")
        for k in k_values:
            print(
                f"  P@{k}: {metrics['precision'][str(k)]:.4f} | "
                f"R@{k}: {metrics['recall'][str(k)]:.4f} | "
                f"nDCG@{k}: {metrics['ndcg'][str(k)]:.4f}"
            )

    print(f"\nDetailed report: {detailed_path}")
    print(f"Summary report: {summary_path}")


if __name__ == "__main__":
    main()
