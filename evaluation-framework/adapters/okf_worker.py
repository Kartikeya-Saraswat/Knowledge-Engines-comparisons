import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2] / "hybrid-rag"
sys.path.insert(0, str(_PROJECT_ROOT))
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src.embeddings.embedder import Embedder
from src.ingestion.okf_loader import load_okf_directory
from src.retrieval.vector_store import VectorStore


def build_index(okf_directory, index_directory):
    okf_directory = Path(okf_directory).resolve()
    index_directory = Path(index_directory).resolve()

    if not okf_directory.is_dir():
        raise FileNotFoundError(f"OKF directory not found: {okf_directory}")

    concepts = load_okf_directory(okf_directory)

    if not concepts:
        raise ValueError(f"No OKF concepts found in: {okf_directory}")

    index_directory.mkdir(parents=True, exist_ok=True)

    embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
    vector_store = VectorStore(persist_directory=index_directory)

    embedded_records = embedder.embed_okf_concepts(concepts)
    indexed_count = vector_store.upsert_okf_concepts(embedded_records)

    return {
        "concept_count": len(concepts),
        "indexed_count": indexed_count,
        "index_directory": str(index_directory),
    }


def convert_and_index(request):
    converter_root = Path(request["converter_directory"]).resolve()
    converter_python = Path(request["converter_python"]).resolve()
    pdf_directory = Path(request["pdf_directory"]).resolve()
    output_directory = Path(request["conversion_output_directory"]).resolve()

    script = converter_root / "scripts" / "run_pipeline.py"

    for path in (converter_python, pdf_directory, script):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    command = [
        str(converter_python),
        str(script),
        "--input-dir",
        str(pdf_directory),
        "--output-dir",
        str(output_directory),
    ]

    started = time.perf_counter()

    process = subprocess.run(
        command,
        cwd=str(converter_root),
        capture_output=True,
        text=True,
        check=False,
    )

    conversion_seconds = time.perf_counter() - started

    if process.returncode != 0:
        raise RuntimeError(
            "PDF-to-OKF conversion failed.\n"
            f"STDOUT:\n{process.stdout[-5000:]}\n"
            f"STDERR:\n{process.stderr[-5000:]}"
        )

    unified_directory = output_directory / "unified"
    report_path = output_directory / "reports" / "pipeline_summary.json"

    if not report_path.exists():
        raise FileNotFoundError(
            f"Conversion report not found: {report_path}"
        )

    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)

    if not report.get("required_checks_passed", False):
        raise RuntimeError(
            "Generated OKF knowledge base did not pass required validation."
        )

    index_result = build_index(
        unified_directory,
        request["index_directory"],
    )

    return {
        "status": "success",
        "operation": "convert_and_index",
        "conversion_seconds": round(conversion_seconds, 3),
        "conversion_report": str(report_path),
        "conversion_pdf_count": report.get("pdf_count"),
        "conversion_concept_count": report.get("concept_count"),
        "required_checks_passed": True,
        "indexing": index_result,
        "converter_stdout": process.stdout[-3000:],
    }


def retrieve(request):
    index_directory = Path(request["index_directory"]).resolve()
    question = request["question"]
    top_k = request.get("top_k", 5)

    if not index_directory.exists():
        raise FileNotFoundError(
            f"OKF evaluation index not found: {index_directory}. "
            "Run indexing first."
        )

    embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
    vector_store = VectorStore(persist_directory=index_directory)

    query_embedding = embedder.embed_query(question)

    records = vector_store.search_okf(
        query_embedding=query_embedding,
        top_k=top_k,
    )

    return {
        "status": "success",
        "arm": "okf_pipeline",
        "question": question,
        "results": [
            {
                "record_id": record.record_id,
                "text": record.text,
                "metadata": record.metadata,
                "distance": record.distance,
                "source_type": "okf",
            }
            for record in records
        ],
    }


def main():
    request = json.load(sys.stdin)
    operation = request["operation"]

    if operation == "index":
        started = time.perf_counter()

        result = build_index(
            request["okf_directory"],
            request["index_directory"],
        )

        result.update({
            "status": "success",
            "operation": "index",
            "indexing_seconds": round(
                time.perf_counter() - started, 3
            ),
        })

    elif operation == "convert_and_index":
        result = convert_and_index(request)

    elif operation == "retrieve":
        result = retrieve(request)

    else:
        raise ValueError(f"Unknown operation: {operation}")

    print(json.dumps(result, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"OKF worker error: {exc}", file=sys.stderr)
        raise
