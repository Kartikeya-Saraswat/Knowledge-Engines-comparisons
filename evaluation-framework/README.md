@'
# Knowledge Retrieval Evaluation Framework

## Objective

A standalone framework to evaluate and compare three knowledge retrieval approaches using a common NCERT textbook corpus.

## Systems Under Evaluation

### A. PDF RAG Baseline
Retrieves relevant text chunks directly from original PDF textbooks using vector similarity search.

### B. OKF-only Retrieval (Ablation)
Retrieves structured OKF concepts using the existing retrieval components in Hybrid RAG.

This is an experimental ablation, not an independently implemented QA system.

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