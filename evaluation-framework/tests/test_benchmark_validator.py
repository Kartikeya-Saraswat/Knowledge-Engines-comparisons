import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.benchmark_validator import (
    load_concept_texts, validate_benchmark, validate_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def example():
    return [{
        "id": "Q1", "answerable": True, "evidence_units": [{
            "id": "E1", "fact": "Pipelines transport solids as slurry.", "relevance": 3,
            "pdf_anchor_mode": "any",
            "pdf_anchors": [{"document": "source.pdf", "pages": [1],
                             "quote": "Pipelines transport solids as slurry."}],
            "okf_concept_ids": ["C1"],
            "okf_support": [{"concept_id": "C1", "support": "full",
                             "quotes": ["Pipelines transport solids as slurry."]}],
        }],
    }]


def test_source_and_support_excerpts_are_checked(example):
    assert validate_benchmark(example, {"C1": "Pipelines transport solids as slurry."},
                              {"source.pdf": ["Pipelines transport solids as slurry."]}) == []


def test_partial_support_cannot_be_promoted_to_full_credit(example):
    unit = example[0]["evidence_units"][0]
    unit["okf_support"][0].update(support="partial", reason="Missing the transport material.")
    errors = validate_benchmark(example, {"C1": "Pipelines transport solids as slurry."})
    assert any("partial support earns no full credit" in error for error in errors)
    unit["okf_concept_ids"] = []
    unit["coverage_note"] = "Only partial support remains."
    assert validate_benchmark(example, {"C1": "Pipelines transport solids as slurry."}) == []


def test_empty_mapping_is_valid_for_lost_knowledge(example):
    unit = example[0]["evidence_units"][0]
    unit.update(okf_concept_ids=[], okf_support=[], coverage_note="Conversion omitted this fact.")
    assert validate_benchmark(example, {}) == []


def test_concept_id_alone_is_not_an_audit(example):
    example[0]["evidence_units"][0]["okf_support"] = []
    errors = validate_benchmark(example, {"C1": "Pipelines transport solids as slurry."})
    assert any("full-support IDs must exactly match" in error for error in errors)


def test_wrong_page_and_invented_concept_excerpt_are_rejected(example):
    errors = validate_benchmark(example, {"C1": "Unrelated concept text."},
                                {"source.pdf": ["Unrelated page text."]})
    assert any("PDF quote not found" in error for error in errors)
    assert any("supporting quote not found" in error for error in errors)


def test_joint_anchors_must_be_split(example):
    example[0]["evidence_units"][0]["pdf_anchor_mode"] = "all"
    assert any("split joint facts" in error for error in validate_benchmark(example, {}))


def test_independently_sufficient_alternative_anchors_are_allowed(example):
    unit = example[0]["evidence_units"][0]
    alternative = copy.deepcopy(unit["pdf_anchors"][0])
    alternative["pages"] = [2]
    unit["pdf_anchors"].append(alternative)
    assert validate_benchmark(example, {"C1": "Pipelines transport solids as slurry."},
                              {"source.pdf": ["Pipelines transport solids as slurry."] * 2}) == []


@pytest.mark.parametrize("quotes", [[], ["missing words"], ["as slurry", "Pipelines"]])
def test_required_matching_excerpts_must_be_complete_and_ordered_in_source(example, quotes):
    example[0]["evidence_units"][0]["pdf_anchors"][0]["required_quotes"] = quotes
    errors = validate_benchmark(example, {"C1": "Pipelines transport solids as slurry."})
    assert any("required PDF excerpts" in error for error in errors)


def test_duplicate_fact_and_question_ids_are_rejected(example):
    duplicate = copy.deepcopy(example[0]["evidence_units"][0])
    duplicate["id"] = "E2"
    example[0]["evidence_units"].append(duplicate)
    example.append(copy.deepcopy(example[0]))
    errors = validate_benchmark(example, {"C1": "Pipelines transport solids as slurry."})
    assert any("duplicate fact" in error for error in errors)
    assert any("duplicate question ID" in error for error in errors)


@pytest.mark.parametrize("pages", [[], [True], [0], [-1], ["1"]])
def test_invalid_physical_pages_are_rejected(example, pages):
    example[0]["evidence_units"][0]["pdf_anchors"][0]["pages"] = pages
    assert any("invalid physical PDF pages" in error for error in validate_benchmark(example, {}))


def test_audited_benchmark_has_consistent_support_and_frozen_inputs():
    questions = json.loads((ROOT / "benchmark/questions.json").read_text(encoding="utf-8"))
    snapshot = json.loads((ROOT / "benchmark/evidence_audit.json").read_text(encoding="utf-8"))
    concepts = ROOT.parent / snapshot["concept_directory"]
    pdfs = ROOT.parent / snapshot["pdf_directory"]
    if not concepts.is_dir() or not pdfs.is_dir():
        pytest.skip("Install the local audited PDF/OKF bundles to run the snapshot integration check")
    assert validate_benchmark(questions, load_concept_texts(concepts)) == []
    assert validate_snapshot(snapshot, ROOT / "benchmark/questions.json", concepts, pdfs) == []
    assert snapshot["question_count"] == len(questions)
    assert snapshot["evidence_unit_count"] == sum(len(q["evidence_units"]) for q in questions)


def test_known_mapping_regressions_and_atomic_waterways():
    questions = {q["id"]: q for q in json.loads(
        (ROOT / "benchmark/questions.json").read_text(encoding="utf-8")
    )}
    ore_units = questions["JESS105-Q07"]["evidence_units"]
    assert all(not unit["okf_concept_ids"] for unit in ore_units
               if "exported" in unit["fact"] or "Kudremukh" in unit["fact"])
    ratio = next(unit for unit in questions["JESS106-Q04"]["evidence_units"] if "4:2:1" in unit["fact"])
    assert set(ratio["okf_concept_ids"]) == {"KC-JESS106-003", "KC-JESS106-005"}
    assert "KC-JESS106-004" not in ratio["okf_concept_ids"]
    waterways = questions["JESS107-Q04"]["evidence_units"]
    assert len([unit for unit in waterways if "listed length" in unit["fact"]]) == 5
    assert all(len(unit["pdf_anchors"]) == 1 for unit in waterways)
    assert all(not unit["okf_concept_ids"] for unit in waterways
               if "Kottapuram" in unit["fact"] or "Kakinada" in unit["fact"])


def test_snapshot_detects_benchmark_changes(tmp_path):
    benchmark = tmp_path / "questions.json"
    benchmark.write_text("[]", encoding="utf-8")
    snapshot = {"schema_version": 2, "benchmark_sha256": hashlib.sha256(b"[]").hexdigest(),
                "concept_files": [], "pdf_files": []}
    assert validate_snapshot(snapshot, benchmark, tmp_path, tmp_path) == []
    benchmark.write_text("[{}]", encoding="utf-8")
    assert "Benchmark differs from the audited revision" in validate_snapshot(
        snapshot, benchmark, tmp_path, tmp_path,
    )


def test_descriptions_count_but_stored_provenance_does_not(tmp_path):
    (tmp_path / "concept.md").write_text(
        "---\nid: C1\ntitle: Pipelines\ndescription: Pipelines carry slurry.\n"
        "evidence: [{quote: Secret source sentence.}]\n---\n\n"
        "Concept body.\n\n## Source provenance\n\nSecret source sentence.\n",
        encoding="utf-8",
    )
    concepts = load_concept_texts(tmp_path)
    assert "Pipelines carry slurry." in concepts["C1"]
    assert "Secret source sentence." not in concepts["C1"]
