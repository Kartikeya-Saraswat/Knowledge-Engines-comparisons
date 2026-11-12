import json
from pathlib import Path

import pytest

from src.evaluation.evaluate import run_evaluation
from src.evaluation.experiment import ARMS, experiment, latency_summary


def question():
    return {"id": "Q1", "type": "factual", "answerable": True, "question": "What is a resource?",
            "evidence_units": [{"id": "E1", "fact": "Resources are useful.", "relevance": 3,
                                "pdf_anchor_mode": "any", "pdf_anchors": [{"document": "a.pdf", "pages": [1],
                                                                           "quote": "Resources are useful."}],
                                "okf_concept_ids": ["C1"], "okf_support": [{"concept_id": "C1", "support": "full",
                                                                          "quotes": ["Resources are useful."]}]}]}


def test_experiment_writes_exact_run_and_excludes_startup_from_latency(tmp_path, monkeypatch):
    import src.evaluation.experiment as module
    def freeze(config, destination):
        destination.mkdir(parents=True)
        (destination/'questions.json').write_text(json.dumps([question()]), encoding='utf-8')
        return [question()], {"version": "test-v1", "status": "benchmark_evidence_reviewed_not_full_content"}
    monkeypatch.setattr(module, "freeze_inputs", freeze)
    calls = []
    def worker(python, script, request, **kwargs):
        calls.append(request)
        is_okf = request['system']=='okf'
        result = {"record_id": "C1" if is_okf else "pdf1", "text": "Resources are useful.",
                  "metadata": {"source_type": "okf" if is_okf else "pdf", "concept_id": "C1",
                               "document_name": "a.pdf", "page_numbers": "[1]"}}
        timings = [dict(embedding=.01, search=.02, fusion=0, total=.03)] * request['repetitions']
        return dict(status='success', counts={'okf': 1} if is_okf else {'pdf': 1},
                    initialization_seconds=99, index_seconds=88, warmup_seconds=77,
                    retrieval_records=[dict(question_id='Q1', arm=request['arm'], results=[result], timings=timings)])
    config = {"_config_directory": str(tmp_path), "outputs": {"results_directory": "results"},
              "evaluation": {"arms": list(ARMS), "retrieval_top_k": [1]},
              "systems": {s: {"python_executable": "python", "project_directory": "."} for s in ARMS.values()},
              "models": {"embedding": {"name": "test", "dimensions": 2}}}
    run = experiment(config, limit=1, worker=worker)
    assert len(calls) == 3
    assert (run/'COMPLETE.json').exists()
    assert not (run/'FAILED.json').exists()
    report = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    for arm in ARMS:
        assert report['by_arm'][arm]['recall']['1'] == 1
        assert report['latency_seconds'][arm]['total']['median'] == .03
    assert json.loads((run/'raw.json').read_text())['question_ids'] == ['Q1']


def test_subset_and_unexpected_arms_are_checked():
    benchmark = [question(), {**question(), 'id': 'Q2'}]
    raw = {'arms': ['a'], 'question_ids': ['Q1'], 'top_k_values': [1],
           'retrieval_records': [{'question_id': 'Q1', 'arm': 'a', 'results': []}]}
    assert len(run_evaluation(benchmark, raw, [1])) == 1
    with pytest.raises(ValueError, match='depth'):
        run_evaluation(benchmark, raw, [2])
    raw['retrieval_records'][0]['arm'] = 'unexpected'
    with pytest.raises(ValueError, match='Unexpected arm'):
        run_evaluation(benchmark, raw, [1])


def test_latency_nearest_rank_percentile():
    assert latency_summary(list(range(1, 21))) == {'samples': 20, 'median': 10.5, 'p95': 19}


def test_pdf_approaches_share_identical_chunk_records():
    import runpy
    from types import SimpleNamespace
    root = Path(__file__).resolve().parents[2]
    baseline = runpy.run_path(str(root/'pdf-rag-baseline/src/chunking/text_chunker.py'))
    hybrid = runpy.run_path(str(root/'hybrid-rag/src/chunking/text_chunker.py'))
    pages = [SimpleNamespace(document_name='a.pdf', page_number=1,
                             text='A first paragraph.\nSECOND SECTION\n' + 'word ' * 400)]
    assert baseline['chunk_pages'](pages) == hybrid['chunk_pages'](pages)
