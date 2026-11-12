"""Exercise concept-only QA without downloading models or calling Ollama."""

import json
from types import SimpleNamespace

import pytest

from src.embeddings.embedder import Embedder, EmbeddedConcept
from src.generation.answer_generator import AnswerGenerator, INSUFFICIENT_EVIDENCE
from src.pipeline import OKFRAGPipeline
from src.retrieval.vector_store import RetrievedConcept, VectorStore

QUOTE = "Resource planning involves identifying and developing available resources."


def write_concept(directory, concept_id="KC-TEST-001", filename="planning.md"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(
        "---\n"
        f"type: KnowledgeConcept\nid: {concept_id}\ntitle: Resource planning\n"
        "description: Planning resources.\nversion: '1.0'\nstatus: stable\n"
        "tags: [planning]\nsources: ['sample.pdf, pages 1']\n"
        "evidence:\n- page_number: 1\n"
        f"  quote: {QUOTE}\n---\n\n# Resource planning\n\n{QUOTE}\n\n"
        "## Source provenance\n\n- PDF: `sample.pdf`\n- Pages: 1\n",
        encoding="utf-8",
    )
    return path


class FakeBackend:
    def passage_embed(self, texts):
        return [[1.0, 0.0] for text in texts]

    def query_embed(self, queries):
        return [[1.0, 0.0] for query in queries]


class FakeClient:
    def __init__(self, answer="Planning identifies available resources. [E1]"):
        self.answer = answer
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=self.answer))


@pytest.fixture
def qa(tmp_path):
    client = FakeClient()
    store = VectorStore(tmp_path / "index")
    pipeline = OKFRAGPipeline(
        embedder=Embedder(backend=FakeBackend()), vector_store=store,
        answer_generator=AnswerGenerator(client=client),
    )
    return pipeline, client


def evidence(source_type="okf"):
    return RetrievedConcept("KC-TEST-001", QUOTE, {
        "source_type": source_type, "title": "Resource planning",
        "source_files": json.dumps(["sample.pdf"]), "source_pages": "[1]",
    }, 0.0)


def test_saved_concept_to_answer_preserves_provenance(qa, tmp_path):
    pipeline, client = qa
    write_concept(tmp_path / "concepts")
    (tmp_path / "concepts" / "index.md").write_text("# Index", encoding="utf-8")
    assert pipeline.index_okf_directory(tmp_path / "concepts") == 1
    result = pipeline.answer_question("What is resource planning?", top_k=8)
    assert len(result.retrieved_concepts) == 1
    assert result.citations[0].record_id == "KC-TEST-001"
    assert result.citations[0].source_files == ("sample.pdf",)
    assert result.citations[0].source_pages == (1,)
    assert json.loads(result.retrieved_concepts[0].metadata["evidence"])[0]["quote"] == QUOTE
    prompt = client.calls[0]["messages"][1]["content"]
    assert QUOTE in prompt
    assert "Source provenance" not in prompt
    assert client.calls[0]["options"]["temperature"] == 0


def test_reindex_removes_obsolete_concepts(qa, tmp_path):
    pipeline, _ = qa
    write_concept(tmp_path / "first", "KC-TEST-001")
    write_concept(tmp_path / "second", "KC-TEST-002")
    pipeline.index_okf_directory(tmp_path / "first")
    pipeline.index_okf_directory(tmp_path / "second")
    assert [item.record_id for item in pipeline.retrieve("planning")] == ["KC-TEST-002"]


def test_duplicate_ids_leave_index_intact(qa, tmp_path):
    pipeline, _ = qa
    directory = tmp_path / "concepts"
    write_concept(directory)
    pipeline.index_okf_directory(directory)
    write_concept(directory, filename="duplicate.md")
    with pytest.raises(ValueError, match="Duplicate"):
        pipeline.index_okf_directory(directory)
    assert pipeline.vector_store.collection.count() == 1


def test_failed_embedding_leaves_previous_corpus(qa, tmp_path, monkeypatch):
    pipeline, _ = qa
    write_concept(tmp_path / "concepts")
    pipeline.index_okf_directory(tmp_path / "concepts")
    monkeypatch.setattr(pipeline.embedder.backend, "passage_embed", lambda texts: [])
    with pytest.raises(RuntimeError, match="vector count"):
        pipeline.index_okf_directory(tmp_path / "concepts")
    assert pipeline.vector_store.collection.count() == 1


def test_empty_index_abstains_without_model_call(qa):
    pipeline, client = qa
    result = pipeline.answer_question("What is resource planning?")
    assert result.answer == INSUFFICIENT_EVIDENCE
    assert not result.citations
    assert not client.calls


@pytest.mark.parametrize("question,top_k", [("", 5), ("planning", 0)])
def test_invalid_queries_rejected(qa, question, top_k):
    pipeline, _ = qa
    with pytest.raises(ValueError):
        pipeline.retrieve(question, top_k=top_k)


@pytest.mark.parametrize("answer,error", [
    ("Unsupported claim [E99]", ValueError),
    ("Uncited claim", ValueError),
    ("", RuntimeError),
])
def test_invalid_answers_rejected(answer, error):
    with pytest.raises(error):
        AnswerGenerator(client=FakeClient(answer)).generate("planning?", [evidence()])


def test_pdf_evidence_rejected():
    client = FakeClient()
    with pytest.raises(ValueError, match="only OKF"):
        AnswerGenerator(client=client).generate("planning?", [evidence("pdf")])
    assert not client.calls


def test_explicit_abstention_is_accepted():
    result = AnswerGenerator(client=FakeClient(INSUFFICIENT_EVIDENCE)).generate(
        "unanswerable?", [evidence()],
    )
    assert result.answer == INSUFFICIENT_EVIDENCE
    assert not result.citations


def test_changed_embedding_model_rejected(qa):
    pipeline, _ = qa
    with pytest.raises(ValueError, match="Embedding model differs"):
        VectorStore("unused", model_name="different-model",
                    client=SimpleNamespace(get_or_create_collection=lambda **kwargs:
                        pipeline.vector_store.collection))


def test_store_rejects_pdf_records(qa):
    pipeline, _ = qa
    with pytest.raises(ValueError, match="only OKF"):
        pipeline.vector_store.replace_concepts([
            EmbeddedConcept("pdf", "raw PDF", {"source_type": "pdf"}, [1.0, 0.0]),
        ])


def test_query_cli_indexes_saved_concepts(qa, tmp_path, monkeypatch, capsys):
    from scripts import query_okf

    pipeline, _ = qa
    write_concept(tmp_path / "concepts")
    monkeypatch.setattr(query_okf, "OKFRAGPipeline", lambda *args, **kwargs: pipeline)
    assert query_okf.main(["--okf-dir", str(tmp_path / "concepts"),
                           "--question", "What is resource planning?"]) == 0
    assert "KC-TEST-001" in capsys.readouterr().out


def test_convert_then_answer_and_failed_validation_gate(qa, tmp_path, monkeypatch, capsys):
    import pymupdf
    from scripts import run_pipeline
    import src.pipeline as pipeline_module

    pipeline, client = qa
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), QUOTE)
        document.save(pdf_dir / "sample.pdf")
    monkeypatch.setattr(run_pipeline, "generate_concepts", lambda pages, source: [{
        "title": "Resource planning", "description": "Planning resources.",
        "tags": ["planning"], "content": QUOTE, "source_pages": [1],
        "evidence": [{"page_number": 1, "quote": QUOTE}],
    }])
    monkeypatch.setattr(pipeline_module, "OKFRAGPipeline", lambda *args, **kwargs: pipeline)
    output = tmp_path / "output"
    args = ["--input-dir", str(pdf_dir), "--output-dir", str(output),
            "--question", "What is resource planning?"]
    assert run_pipeline.main(args) == 0
    assert len(client.calls) == 1
    assert (output / "unified" / "index.md").is_file()
    original = (output / "unified" / "sample__resource-planning.md").read_bytes()
    # A second question must reuse both conversion output and passage embeddings.
    with monkeypatch.context() as cached:
        def unexpected(*args, **kwargs):
            raise AssertionError("Unchanged inputs must not regenerate or re-embed")
        cached.setattr(run_pipeline, "generate_concepts", unexpected)
        cached.setattr(run_pipeline, "build_page_plan", unexpected)
        cached.setattr(pipeline.embedder.backend, "passage_embed", unexpected)
        assert run_pipeline.main(args) == 0
        assert len(client.calls) == 2
        assert run_pipeline.main(args + ["--dry-run"]) == 0
        assert len(client.calls) == 2
    monkeypatch.setattr(run_pipeline, "validate_okf_directory",
                        lambda directory: {"concept.md": ["forced validation failure"]})
    assert run_pipeline.main(args + ["--force-rebuild"]) == 1
    assert len(client.calls) == 2
    assert (output / "unified" / "sample__resource-planning.md").read_bytes() == original


@pytest.mark.parametrize("change", ["pdf", "settings", "concepts"])
def test_changed_inputs_do_not_reuse_conversion(tmp_path, monkeypatch, change):
    from scripts import run_pipeline
    from src.ingestion.fingerprints import fingerprint_files, fingerprint_concepts
    import os

    input_dir = tmp_path / "pdfs"
    input_dir.mkdir()
    pdf = input_dir / "sample.pdf"
    pdf.write_bytes(b"fingerprinted input")
    output = tmp_path / "output"
    concept = write_concept(output / "unified")
    signature = {
        "version": 1, "pdf_fingerprint": fingerprint_files([pdf], input_dir),
        "model": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        "batch_size": 3, "max_pages": None, "max_batches": None,
        "link": False, "max_related": 3,
    }
    reports = output / "reports"
    reports.mkdir()
    (reports / "pipeline_summary.json").write_text(json.dumps({
        "conversion_signature": signature, "required_checks_passed": True,
        "concept_fingerprint": fingerprint_concepts(output / "unified"),
    }), encoding="utf-8")
    args = ["--input-dir", str(input_dir), "--output-dir", str(output)]
    if change == "pdf":
        pdf.write_bytes(b"changed input")
    elif change == "settings":
        args += ["--max-pages", "1"]
    else:
        concept.write_text(concept.read_text(encoding="utf-8") + "\nChanged text.\n", encoding="utf-8")

    def reached_conversion(*args, **kwargs):
        raise RuntimeError("Conversion was requested")
    monkeypatch.setattr(run_pipeline, "build_page_plan", reached_conversion)
    with pytest.raises(RuntimeError, match="Conversion was requested"):
        run_pipeline.main(args)


def test_dry_run_does_not_create_output_or_answer(tmp_path, monkeypatch):
    from scripts import run_pipeline

    input_dir = tmp_path / "pdfs"
    input_dir.mkdir()
    (input_dir / "sample.pdf").write_bytes(b"placeholder")
    monkeypatch.setattr(run_pipeline, "build_page_plan", lambda *args, **kwargs: {"sample": 1})
    output = tmp_path / "output"
    assert run_pipeline.main(["--input-dir", str(input_dir), "--output-dir", str(output),
                              "--question", "planning?", "--dry-run"]) == 0
    assert not output.exists()


def test_evaluation_worker_indexes_and_retrieves_with_converter(qa, tmp_path, monkeypatch):
    import runpy
    from pathlib import Path

    pipeline, client = qa
    write_concept(tmp_path / "concepts")
    worker = runpy.run_path(str(Path(__file__).resolve().parents[2]
                               / "evaluation-framework/adapters/okf_worker.py"))
    monkeypatch.setitem(worker["build_index"].__globals__, "OKFRAGPipeline",
                        lambda **kwargs: pipeline)
    index_dir = tmp_path / "index"
    indexed = worker["build_index"](tmp_path / "concepts", index_dir)
    assert indexed["indexed_count"] == 1
    result = worker["retrieve"]({
        "index_directory": str(index_dir), "question": "planning?", "top_k": 8,
        "arm": "okf_only_ablation",
    })
    assert result["arm"] == "okf_only_ablation"
    assert result["results"][0]["source_type"] == "okf"
    assert result["results"][0]["record_id"] == "KC-TEST-001"
    assert not client.calls


def test_reopened_index_can_answer_without_loading_concept_files(qa, tmp_path):
    pipeline, _ = qa
    write_concept(tmp_path / "concepts")
    pipeline.index_okf_directory(tmp_path / "concepts")
    reopened = OKFRAGPipeline(
        tmp_path / "index", embedder=Embedder(backend=FakeBackend()),
        answer_generator=AnswerGenerator(client=FakeClient()),
    )
    assert reopened.answer_question("planning?").citations[0].record_id == "KC-TEST-001"


def test_concept_changes_refresh_index_but_unchanged_files_reuse_it(qa, tmp_path, monkeypatch):
    pipeline, _ = qa
    path = write_concept(tmp_path / "concepts")
    calls = []
    original = pipeline.embedder.backend.passage_embed
    monkeypatch.setattr(pipeline.embedder.backend, "passage_embed",
                        lambda texts: (calls.append(texts), original(texts))[1])
    pipeline.ensure_okf_directory(path.parent)
    pipeline.ensure_okf_directory(path.parent)
    assert len(calls) == 1
    path.write_text(path.read_text(encoding="utf-8") + "\nAn additional fact.\n", encoding="utf-8")
    pipeline.ensure_okf_directory(path.parent)
    assert len(calls) == 2


def test_legacy_bundle_requires_explicit_reuse(qa, tmp_path, monkeypatch):
    from scripts import run_pipeline
    import src.pipeline as pipeline_module

    pipeline, client = qa
    output = tmp_path / "legacy"
    write_concept(output / "unified")
    input_dir = tmp_path / "pdfs"
    input_dir.mkdir()
    (input_dir / "sample.pdf").write_bytes(b"PDF fingerprint input")
    monkeypatch.setattr(pipeline_module, "OKFRAGPipeline", lambda *args, **kwargs: pipeline)
    args = ["--input-dir", str(input_dir), "--output-dir", str(output),
            "--question", "planning?"]
    with pytest.raises(SystemExit, match="2"):
        run_pipeline.main(args)
    assert not client.calls
    # Explicit reuse does not even require the original PDFs to remain available.
    args[1] = str(tmp_path / "missing-pdfs")
    assert run_pipeline.main(args + ["--reuse-concepts"]) == 0
    assert len(client.calls) == 1


def test_unfinished_staging_is_not_reused(tmp_path):
    from scripts import run_pipeline

    output = tmp_path / "output"
    write_concept(output / ".unified_staging")
    with pytest.raises(SystemExit, match="2"):
        run_pipeline.main(["--output-dir", str(output), "--reuse-concepts",
                           "--question", "planning?"])
