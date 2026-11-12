# PDF to OKF question answering

This project converts PDFs into validated OKF concept Markdown, embeds those
concepts, retrieves relevant concepts, and generates cited answers with Ollama.
Retrieval and answer generation use only concept content. Original PDF names and
physical page numbers are retained as provenance; QA does not read the PDF text.

Install from this directory with `.venv\Scripts\python.exe -m pip install -r requirements.txt`.
Copy `.env.example` to `.env` and make the selected conversion and answer models
available in the running Ollama service. The embedding model is
`BAAI/bge-small-en-v1.5`; its first use may download model files.

## Convert PDFs and answer

Run from `pdf-to-okf-auto`:

```powershell
.venv\Scripts\python.exe scripts/run_pipeline.py --input-dir data/input/sample --output-dir data/output/qa_run --question "What is resource planning in India?"
```

Use `--interactive` to ask multiple questions after conversion. `--top-k` controls
the number of concepts retrieved. `--answer-model` overrides `OKF_ANSWER_MODEL`.
The default QA index is `OUTPUT/chroma_db`; override it with `--index-dir`.
Completed runs save fingerprints of the input PDFs, conversion settings, and
concept files. Repeating the same command reuses the concepts and embeddings;
only retrieval and answer generation run again. Changes to those inputs trigger
conversion again. Use `--force-rebuild` to regenerate concepts explicitly.
QA starts only after required conversion checks pass. `--dry-run` makes no model
calls or index writes, including when a completed run is reusable.

For older completed bundles created before fingerprint tracking, use
`--reuse-concepts` to use `OUTPUT/unified` without reconverting or checking the
original PDFs. For example:

```powershell
.venv\Scripts\python.exe scripts/run_pipeline.py --output-dir data/output/ollama_full_run --reuse-concepts --question "What is resource planning?"
```

Existing untracked bundles are never silently regenerated: the command requests
either explicit reuse or `--force-rebuild`. An unfinished `.unified_staging`
directory is not a completed corpus and cannot be reused for QA.

## Query saved concepts without converting again

```powershell
.venv\Scripts\python.exe scripts/query_okf.py --okf-dir data/output/qa_run/unified --index-dir data/output/qa_run/chroma_db --question "What is resource planning in India?"
```

This reuses embeddings when the concept files are unchanged and synchronizes the
index when they change, including removing obsolete concepts. Omit `--question`
for an interactive session. Add `--no-index` to query the stored index without
checking the concept files. Relative paths resolve from this project directory,
regardless of the working directory.

Answers use evidence markers such as `[E1]`, followed by concept IDs, titles, and
PDF provenance. Invalid evidence IDs and uncited answers are rejected. Citation
validation checks identifiers, not whether every generated claim is entailed.
No-evidence retrieval returns an insufficient-evidence answer.

Existing concept bundles can be queried even if they predate stored evidence
quotes. That does not make them validated outputs of the current converter;
the known quality limitations of older bundles still apply.

## Retrieval evaluation

The evaluation framework's OKF arm now runs `adapters/okf_worker.py` using this
project's Python environment. Hybrid RAG keeps its own PDF and OKF functionality.
Build the dedicated evaluation index from the repository root:

```powershell
'{"operation":"index","okf_directory":"pdf-to-okf-auto/data/output/ollama_full_run/unified","index_directory":"evaluation-framework/data/indexes/okf"}' | .\pdf-to-okf-auto\.venv\Scripts\python.exe evaluation-framework/adapters/okf_worker.py
```

Then use the evaluation runner as before. Its `okf_only_ablation` arm now selects
this standalone pipeline. Retrieval evaluation does not call the answer model.
Historical results still describe the implementation that produced them.
