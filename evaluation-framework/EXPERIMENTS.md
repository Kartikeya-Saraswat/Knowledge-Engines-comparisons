# Controlled three-way retrieval experiments

From `evaluation-framework`, install requirements in its environment, then run:

```powershell
.\.venv\Scripts\python.exe -B -m src.evaluation.experiment --repetitions 3
```

Use `--limit 3` for a small end-to-end check. Models must be available in each
configured project environment; first use may download embeddings. Ollama is not
called. `config.yaml` controls the common embedding model/dimensions, PDF chunk
size/overlap, RRF constant, corpus paths and expected counts. All three arms run.

Each unique `results/RUN_ID` contains frozen input PDFs and original OKF Markdown,
benchmark/evidence audit, a versioned `corpus_validation.json` provenance sidecar,
new isolated indexes, manifest, raw rankings/repeat timings, detailed per-question
metrics, summaries by question type, latency distributions, and a Markdown report.
`COMPLETE.json` is written only after scoring succeeds. Failures retain diagnostic
artifacts with `FAILED.json`; previous corpora, indexes and reports are untouched.
The manifest records source hashes (including uncommitted source), Git revision,
resolved configuration, worker Python/dependency versions, index bytes/counts,
initialization/indexing/warmup times, corpus version and selected question IDs.

## Metrics, version 2

- Precision@K: relevant returned records divided by K, including repeated relevant
  records. Missing result slots count as nonrelevant.
- Evidence Recall@K: distinct matched facts divided by required facts. Repeated
  evidence earns no extra recall.
- Complete Success@K: 1 when all required facts were retrieved, otherwise 0;
  undefined for questions without evidence.
- MRR@max(K): reciprocal first supporting rank within the collected depth,
  macro-averaged over questions. This does not measure answer completeness.
- Duplicate rate@K: repeated evidence hits divided by all evidence hits; 0 when
  no evidence was matched. One record may hit several facts.
- nDCG and MAP: excluded (`null`), since record-level relevance ideals have not
  been established across these different representations.

Worker processes load once, build indexes, and warm up before timing questions.
Timing covers query embedding, vector search, fusion, and their total; it excludes
process startup, model loading, indexing, warmup and response serialization.
Summaries report median and nearest-rank p95 across repeated queries. Hybrid
searches K candidates in each collection before returning K; single-source arms
search one collection. Repeat rankings are saved to expose nondeterminism.

Both PDF approaches use `retrieval_shared/pdf_chunking.py` and, in experiments,
the same PDF loader and explicit embedding parameters. This replaces the earlier
different chunkers. Rebuild indexes before comparing; old scores are incompatible.

## Corpus review boundary

The freeze preserves the audited benchmark facts and all 103 original concepts.
Evidence lives in a sidecar and is NEVER appended to retrieval text (which would
leak benchmark labels). Review status is explicitly
`benchmark_evidence_reviewed_not_full_content`. Some generated concepts contain
known factual contamination; the earlier evidence audit is not a semantic review
of every claim. Full content correction/review remains editorial work before a
claim of a fully validated knowledge base. No missing quotations are fabricated.
Conversion cost is offline and is not included in retrieval timing; historical
conversion duration is unavailable for this corpus and must not be invented.

The 27-question benchmark is a pilot. The reports are descriptive, not statistical
proof of superiority. Held-out questions, paired confidence intervals and manual
failure analysis remain useful research extensions.
