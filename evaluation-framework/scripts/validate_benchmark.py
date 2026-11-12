"""Check evidence annotations, source excerpts, and frozen audit fingerprints."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config, resolve_path
from src.evaluation.benchmark_validator import (
    load_concept_texts, validate_benchmark, validate_snapshot,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    benchmark = resolve_path(config["benchmark"]["file"], config)
    concepts = resolve_path(config["dataset"]["okf_directory"], config)
    pdf_directory = resolve_path(config["dataset"]["pdf_directory"], config)
    snapshot_path = benchmark.parent / "evidence_audit.json"
    questions = json.loads(benchmark.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    # Run this CLI in the converter environment, which already includes PyMuPDF.
    import pymupdf

    pdf_pages = {}
    for path in sorted(pdf_directory.glob("*.pdf")):
        with pymupdf.open(path) as document:
            pdf_pages[path.name] = [page.get_text("text") for page in document]
    concept_texts = load_concept_texts(concepts)
    errors = validate_benchmark(questions, concept_texts, pdf_pages)
    errors += validate_snapshot(snapshot, benchmark, concepts, pdf_directory)
    units = [unit for question in questions for unit in question["evidence_units"]]
    print(f"Questions: {len(questions)}; evidence facts: {len(units)}; concepts checked: {len(concept_texts)}")
    print(f"Facts without full OKF support: {sum(not unit['okf_concept_ids'] for unit in units)}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print("Evidence structure, source excerpts, support excerpts, and frozen fingerprints PASSED.")
    print("Semantic support remains an editorial judgment. Retrieval scoring is a separate check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
