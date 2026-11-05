import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2] / "hybrid-rag"
sys.path.insert(0, str(_PROJECT_ROOT))
import json
import sys
from pathlib import Path

from src.pipeline import HybridRAGPipeline


def serialize_record(record, source_type):
    return {
        "record_id": record.record_id,
        "text": record.text,
        "metadata": record.metadata,
        "distance": getattr(record, "distance", None),
        "rrf_score": getattr(record, "rrf_score", None),
        "source_ranks": getattr(record, "source_ranks", None),
        "source_type": source_type,
    }


def main():
    request = json.load(sys.stdin)

    index_dir = Path(request["index_directory"]).resolve()
    pipeline = HybridRAGPipeline(persist_directory=index_dir)

    operation = request["operation"]

    if operation == "index":
        pdf_dir = Path(request["pdf_directory"]).resolve()
        okf_dir = Path(request["okf_directory"]).resolve()

        counts = pipeline.rebuild_index(
            pdf_directory=pdf_dir,
            okf_directory=okf_dir,
        )

        result = {
            "status": "success",
            "counts": counts,
            "index_directory": str(index_dir),
        }

    elif operation == "retrieve":
        question = request["question"]
        top_k = request.get("top_k", 5)
        arm = request["arm"]

        if arm == "okf_only_ablation":
            records = pipeline.retrieve_okf(question, top_k=top_k)
            results = [
                serialize_record(record, "okf")
                for record in records
            ]

        elif arm == "hybrid_rag":
            records = pipeline.retrieve_hybrid(
                question,
                pdf_top_k=top_k,
                okf_top_k=top_k,
                final_top_k=top_k,
            )
            results = [
                serialize_record(record, "hybrid")
                for record in records
            ]

        elif arm == "hybrid_pdf_only":
            records = pipeline.retrieve_pdf(question, top_k=top_k)
            results = [
                serialize_record(record, "pdf")
                for record in records
            ]

        else:
            raise ValueError(f"Unknown retrieval arm: {arm}")

        result = {
            "status": "success",
            "arm": arm,
            "results": results,
        }

    else:
        raise ValueError(f"Unknown operation: {operation}")

    print(json.dumps(result, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Worker error: {exc}", file=sys.stderr)
        raise
