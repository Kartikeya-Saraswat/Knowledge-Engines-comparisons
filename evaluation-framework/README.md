@'
# Knowledge Retrieval Evaluation Framework

For controlled experiments, use the [single-run workflow](EXPERIMENTS.md):
`python -m src.evaluation.experiment --repetitions 3`.
It freezes inputs, builds fresh indexes, warms persistent workers and writes the
exact run's metrics, timing and reproducibility reports. The older `runner.py`
path remains a legacy retrieval collector; its subprocess times are not pure
retrieval latency. nDCG and MAP are excluded from metric version 2.

## Objective

A standalone framework to evaluate and compare three knowledge retrieval approaches using a common NCERT textbook corpus.

## Systems Under Evaluation

### A. PDF RAG Baseline
Retrieves relevant text chunks directly from original PDF textbooks using vector similarity search.

### B. OKF-only Retrieval (Ablation)
Retrieves structured OKF concepts through the standalone `pdf-to-okf-auto` pipeline.
The historical evaluation arm name `okf_only_ablation` is retained for report compatibility.
It uses its own converter environment and `data/indexes/okf` index. Build this index
with `adapters/okf_worker.py`'s `index` operation before running evaluation;
old indexes created with Hybrid RAG's collections must be rebuilt.

### C. Hybrid RAG
Retrieves from both PDF chunks and OKF concepts and combines their rankings using Reciprocal Rank Fusion (RRF).

## Dataset

- 9 NCERT PDF textbooks
- 100 PDF pages
- 103 OKF concept Markdown files
- 1 OKF index file

## Architecture

The evaluation framework runs the existing systems through isolated subprocess adapters.

Each system retains its own Python environment and implementation.

The evaluator is responsible for:
- Benchmark question management
- Ground-truth evidence management
- Retrieval result collection
- Retrieval metrics
- Answer quality evaluation
- Performance measurement
- Experiment reporting

## Planned Evaluation Metrics

### Retrieval
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG)

### Answer Quality
- Correctness
- Faithfulness to retrieved evidence
- Answer relevance
- Citation correctness
- Appropriate abstention

### Performance
- Index build time
- Retrieval latency
- Generation latency
- End-to-end latency
- Index storage size

PDF-to-OKF conversion time will be measured separately from retrieval indexing.

## Project Structure

```text
evaluation-framework/
├── adapters/
├── benchmark/
├── data/
│   └── indexes/
│       ├── baseline/
│       └── hybrid/
├── results/
├── src/
│   ├── common/
│   ├── evaluation/
│   └── metrics/
├── tests/
├── config.yaml
├── requirements.txt
└── README.md
