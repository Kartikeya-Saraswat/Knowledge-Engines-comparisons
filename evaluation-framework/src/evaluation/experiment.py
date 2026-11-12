"""Validate, freeze, index, retrieve and evaluate all three approaches in one run."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import uuid

from src.common.config import load_config, resolve_path
from src.common.worker import run_worker
from src.evaluation.benchmark_validator import load_concept_texts, validate_benchmark, validate_snapshot
from src.evaluation.evaluate import run_evaluation, build_summaries

ROOT = Path(__file__).resolve().parents[2]
ARMS = {"pdf_rag_baseline": "baseline", "okf_only_ablation": "okf", "hybrid_rag": "hybrid"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def latency_summary(values):
    ordered = sorted(values)
    if not ordered:
        return None
    # Nearest-rank p95, documented consistently for small samples.
    import math
    return dict(samples=len(values), median=statistics.median(values),
                p95=ordered[max(0, math.ceil(.95 * len(ordered))-1)])


def freeze_inputs(config, destination):
    """Freeze original concept bytes and audited evidence without leaking labels into retrieval text."""
    pdf_root = resolve_path(config["dataset"]["pdf_directory"], config)
    okf_root = resolve_path(config["dataset"]["okf_directory"], config)
    benchmark = resolve_path(config["benchmark"]["file"], config)
    audit = json.loads((benchmark.parent / "evidence_audit.json").read_text(encoding="utf-8"))
    questions = json.loads(benchmark.read_text(encoding="utf-8"))
    errors = validate_snapshot(audit, benchmark, okf_root, pdf_root)
    concepts = load_concept_texts(okf_root)
    errors += validate_benchmark(questions, concepts)
    from pypdf import PdfReader
    pages = sum(len(PdfReader(path).pages) for path in pdf_root.glob("*.pdf"))
    expected = config["dataset"]
    if len(list(pdf_root.glob("*.pdf"))) != expected["expected_pdf_count"] or pages != expected["expected_pdf_pages"]:
        errors.append("PDF dataset counts differ from configuration")
    if len(concepts) != expected["expected_okf_concepts"]:
        errors.append("OKF count differs from configuration")
    if errors:
        raise ValueError("Input validation failed: " + "; ".join(errors))
    # Fresh directory owned by this run; never overwrite the canonical corpus.
    destination.mkdir(parents=True)
    shutil.copytree(okf_root, destination / "okf")
    shutil.copytree(pdf_root, destination / "pdf")
    shutil.copy2(benchmark, destination / "questions.json")
    shutil.copy2(benchmark.parent / "evidence_audit.json", destination / "evidence_audit.json")
    provenance = defaultdict(list)
    for q in questions:
        for unit in q["evidence_units"]:
            for support in unit["okf_support"]:
                if support["support"] == "full":
                    provenance[support["concept_id"]].append(dict(
                        question_id=q["id"], evidence_unit_id=unit["id"], fact=unit["fact"],
                        concept_excerpts=support["quotes"], pdf_anchors=unit["pdf_anchors"],
                    ))
    version = digest(benchmark)[:12] + "-" + hashlib.sha256(
        json.dumps(audit["concept_files"], sort_keys=True).encode()).hexdigest()[:12]
    report = dict(version=version, status="benchmark_evidence_reviewed_not_full_content",
                  limitation="Original generated concepts contain known factual contamination; full semantic review is still required.",
                  concepts={key: dict(evidence=provenance[key],
                                      review_scope="benchmark evidence only" if provenance[key] else "no benchmark fact review")
                            for key in concepts},
                  benchmark_sha256=digest(benchmark), source_audit=audit)
    # Sidecar lives outside the Markdown corpus so it cannot influence retrieval.
    write_json(destination / "corpus_validation.json", report)
    return questions, report


def experiment(config, *, limit=None, repetitions=3, worker=run_worker):
    if repetitions < 1 or (limit is not None and limit < 1):
        raise ValueError("Repetitions and limit must be positive")
    ks = config["evaluation"]["retrieval_top_k"]
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("K values must be positive integers")
    arms = config["evaluation"]["arms"]
    if set(arms) != set(ARMS) or len(arms) != 3:
        raise ValueError("A controlled experiment must contain all three distinct approaches")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = resolve_path(config["outputs"]["results_directory"], config) / run_id
    directory.mkdir(parents=True)
    try:
        questions, validation = freeze_inputs(config, directory / "inputs")
        questions = questions[:limit] if limit else questions
        manifest = dict(run_id=run_id, config=config, question_ids=[q["id"] for q in questions],
                        corpus_version=validation["version"], review_status=validation["status"],
                        benchmark_sha256=digest(directory / "inputs/questions.json"),
                        metric_version=2, repetitions=repetitions, arms=arms, workers={})
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True, text=True)
        manifest["git_revision"] = git.stdout.strip()
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=ROOT.parent, capture_output=True)
        manifest["working_diff_sha256"] = hashlib.sha256(diff.stdout).hexdigest()
        # Include uncommitted/untracked first-party source in the reproducibility fingerprint.
        projects = [ROOT, ROOT.parent/'pdf-rag-baseline', ROOT.parent/'hybrid-rag', ROOT.parent/'pdf-to-okf-auto']
        folders = [ROOT.parent/'retrieval_shared'] + [project/sub for project in projects for sub in ['src','scripts','tests','adapters']]
        source_files = [p for folder in folders for p in folder.rglob('*.py') if '__pycache__' not in p.parts]
        manifest["source_hashes"] = {str(p.relative_to(ROOT.parent)): digest(p) for p in source_files}
        preprocessing = config.get("preprocessing", {})
        records = []
        for arm in arms:
            print(f"Building and measuring {arm}...", flush=True)
            system = ARMS[arm]
            settings = config["systems"][system]
            request = dict(system=system, arm=arm, questions=questions, repetitions=repetitions,
                           index_directory=str(directory / "indexes" / system),
                           pdf_directory=str(directory / "inputs/pdf"), okf_directory=str(directory / "inputs/okf"),
                           embedding_model=config["models"]["embedding"]["name"],
                           dimensions=config["models"]["embedding"]["dimensions"], top_k=max(ks),
                           chunk_size=preprocessing.get("chunk_size", 275),
                           chunk_overlap=preprocessing.get("chunk_overlap", 50),
                           rrf_k=preprocessing.get("rrf_k", 60))
            response = worker(resolve_path(settings["python_executable"], config),
                              ROOT / "adapters/experiment_worker.py", request,
                              cwd=resolve_path(settings["project_directory"], config), timeout=3600)
            if response.get("status") != "success":
                raise ValueError(f"Worker failed for {arm}")
            records.extend(response.pop("retrieval_records"))
            response["index_bytes"] = sum(p.stat().st_size for p in (directory/'indexes'/system).rglob('*') if p.is_file())
            manifest["workers"][arm] = response
            write_json(directory / "manifest.json", manifest)
        if manifest["workers"]["pdf_rag_baseline"]["counts"].get("pdf") != manifest["workers"]["hybrid_rag"]["counts"].get("pdf"):
            raise ValueError("Controlled PDF indexes have different chunk counts")
        if manifest['workers']['pdf_rag_baseline'].get('pdf_chunk_sha256') != manifest['workers']['hybrid_rag'].get('pdf_chunk_sha256'):
            raise ValueError("Controlled PDF chunk identities, text or page provenance differ")
        raw = dict(arms=arms, question_ids=manifest["question_ids"], top_k_values=ks, retrieval_records=records)
        write_json(directory / "raw.json", raw)
        detailed = run_evaluation(questions, raw, ks)
        overall, by_type = build_summaries(detailed, ks)
        timing = {}
        for arm in arms:
            timing[arm] = {stage: latency_summary([t[stage] for r in records if r['arm']==arm for t in r['timings']])
                           for stage in ["embedding", "search", "fusion", "total"]}
        if any(digest(ROOT.parent / name) != expected for name, expected in manifest['source_hashes'].items()):
            raise ValueError("Source files changed during the experiment; rerun with stable code")
        drift = validate_snapshot(validation['source_audit'], directory/'inputs/questions.json',
                                  directory/'inputs/okf', directory/'inputs/pdf') if 'source_audit' in validation else []
        if drift:
            raise ValueError("Frozen inputs changed during experiment: " + '; '.join(drift))
        write_json(directory / "detailed.json", detailed)
        write_json(directory / "summary.json", dict(by_arm=overall, by_question_type=by_type, latency_seconds=timing))
        lines = ["# Retrieval experiment " + run_id, "", "Corpus review: " + validation["status"], "",
                 "| Approach | MRR@%s | Recall@%s | Complete@%s | Median seconds | p95 seconds |" % (max(ks), max(ks), max(ks)),
                 "|---|---:|---:|---:|---:|---:|"]
        for arm in arms:
            m = overall[arm]; latency = timing[arm]['total']; key = str(max(ks))
            lines.append(f"| {arm} | {m['mrr']:.4f} | {m['recall'][key]} | {m['complete_success'][key]} | {latency['median']:.4f} | {latency['p95']:.4f} |")
        lines += ["", "nDCG and MAP are excluded. Times exclude initialization, index building and warmup.",
                  "This is a 27-question pilot (or the recorded subset); no general superiority claim is established."]
        (directory / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        write_json(directory / "COMPLETE.json", {"run_id": run_id})
        return directory
    except Exception as exc:
        write_json(directory / "FAILED.json", {"error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    print(experiment(load_config(args.config), limit=args.limit, repetitions=args.repetitions))


if __name__ == "__main__":
    main()
