import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import load_config, resolve_path
from src.common.worker import run_worker

ROOT = Path(__file__).resolve().parents[2]

ARM_WORKERS = {
    "pdf_rag_baseline": ("baseline", "baseline_worker.py"),
    "okf_only_ablation": ("hybrid", "hybrid_worker.py"),
    "hybrid_rag": ("hybrid", "hybrid_worker.py"),
}


def load_benchmark(path):
    with path.open(encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list) or not questions:
        raise ValueError("Benchmark must contain a non-empty question list.")

    return questions


def check_index(index_directory):
    if not index_directory.is_dir():
        raise FileNotFoundError(
            f"Evaluation index not found: {index_directory}. "
            "Build the index before running retrieval."
        )

    if not (index_directory / "chroma.sqlite3").exists():
        raise FileNotFoundError(
            f"ChromaDB database not found in {index_directory}"
        )


def execute_retrieval(config, questions, arms):
    top_k_values = config["evaluation"]["retrieval_top_k"]

    if not top_k_values or any(type(k) is not int or k < 1 for k in top_k_values):
        raise ValueError("retrieval_top_k must contain positive integers.")

    max_top_k = max(top_k_values)
    adapter_directory = ROOT / "adapters"
    records = []

    for arm in arms:
        if arm not in ARM_WORKERS:
            raise ValueError(f"Unsupported evaluation arm: {arm}")

        system_key, worker_filename = ARM_WORKERS[arm]
        system = config["systems"][system_key]

        python_executable = resolve_path(
            system["python_executable"], config
        )
        project_directory = resolve_path(
            system["project_directory"], config
        )
        index_directory = resolve_path(
            system["index_directory"], config
        )
        worker_script = adapter_directory / worker_filename

        if not python_executable.is_file():
            raise FileNotFoundError(f"Python executable not found: {python_executable}")

        if not worker_script.is_file():
            raise FileNotFoundError(f"Worker script not found: {worker_script}")

        check_index(index_directory)

        for question in questions:
            started = time.perf_counter()

            request = {
                "operation": "retrieve",
                "index_directory": str(index_directory),
                "question": question["question"],
                "top_k": max_top_k,
            }

            if arm != "pdf_rag_baseline":
                request["arm"] = arm

            response = run_worker(
                python_executable=python_executable,
                worker_script=worker_script,
                request=request,
                cwd=project_directory,
            )

            elapsed = time.perf_counter() - started

            if response.get("status") != "success":
                raise RuntimeError(
                    f"Retrieval failed for {question['id']} / {arm}: {response}"
                )

            retrieved = response.get("results", [])

            records.append({
                "question_id": question["id"],
                "question": question["question"],
                "question_type": question["type"],
                "arm": arm,
                "requested_top_k": max_top_k,
                "retrieval_seconds": round(elapsed, 4),
                "results": retrieved,
            })

            print(
                f"{arm:22} | {question['id']:15} | "
                f"{len(retrieved)} results | {elapsed:.2f}s"
            )

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Run configurable knowledge retrieval experiments."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    benchmark_path = resolve_path(
        config["benchmark"]["file"], config
    )
    questions = load_benchmark(benchmark_path)

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive.")
        questions = questions[:args.limit]

    arms = args.arms or config["evaluation"]["arms"]

    print(f"Questions: {len(questions)}")
    print(f"Evaluation arms: {', '.join(arms)}")
    print(f"Top-k values: {config['evaluation']['retrieval_top_k']}")

    records = execute_retrieval(config, questions, arms)

    output_directory = resolve_path(
        config["outputs"]["results_directory"], config
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_directory / f"raw_retrieval_{timestamp}.json"

    output = {
        "experiment": config["project"]["name"],
        "created_at": timestamp,
        "benchmark": str(benchmark_path),
        "question_count": len(questions),
        "arms": arms,
        "top_k_values": config["evaluation"]["retrieval_top_k"],
        "retrieval_records": records,
    }

    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"\nRetrieval records: {len(records)}")
    print(f"Results saved: {output_path}")


if __name__ == "__main__":
    main()
