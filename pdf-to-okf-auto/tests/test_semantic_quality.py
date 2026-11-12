import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pipeline import (
    can_promote_knowledge_base,
    collect_grounding_failures,
    collect_required_failures,
)
from src.validation.semantic_quality import analyze_semantic_quality

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.semantic_quality import analyze_semantic_quality


def write_concept(path: Path, concept_id: str, title: str, body: str):
    path.write_text(
        f"""---
type: KnowledgeConcept
id: {concept_id}
title: {title}
description: Test concept.
version: "1.0"
status: stable
tags:
  - test
sources:
  - sample.pdf, pages 1
---

# {title}

{body}

## Source provenance

- PDF: `sample.pdf`
- Pages: 1
""",
        encoding="utf-8",
    )


def test_grounded_concept_has_no_low_overlap_flag(tmp_path):
    okf_dir = tmp_path / "okf"
    okf_dir.mkdir()

    write_concept(
        okf_dir / "resources.md",
        "KC-TEST-001",
        "Natural resources",
        "Natural resources include land, water, soil, forests and minerals.",
    )

    extraction_path = tmp_path / "sample.json"
    extraction_path.write_text(
        json.dumps({
            "source_file": "sample.pdf",
            "pages": [{
                "page_number": 1,
                "text": (
                    "Natural resources include land, water, soil, "
                    "forests and minerals."
                ),
            }],
        }),
        encoding="utf-8",
    )

    report = analyze_semantic_quality(okf_dir, extraction_path)

    assert report["concept_count"] == 1
    assert report["grounding_flags"] == []
    assert report["duplicate_candidates"] == []


def test_similar_concepts_are_flagged_as_duplicates(tmp_path):
    okf_dir = tmp_path / "okf"
    okf_dir.mkdir()

    body = (
        "Natural resources include land water soil forests minerals "
        "and support human activities and economic development."
    )

    write_concept(
        okf_dir / "resources_a.md",
        "KC-TEST-001",
        "Natural resources",
        body,
    )
    write_concept(
        okf_dir / "resources_b.md",
        "KC-TEST-002",
        "Natural resources",
        body,
    )

    extraction_path = tmp_path / "sample.json"
    extraction_path.write_text(
        json.dumps({
            "source_file": "sample.pdf",
            "pages": [{
                "page_number": 1,
                "text": body,
            }],
        }),
        encoding="utf-8",
    )

    report = analyze_semantic_quality(okf_dir, extraction_path)

    assert len(report["duplicate_candidates"]) == 1
    assert report["duplicate_candidates"][0]["similarity"] >= 0.60

def test_collect_grounding_failures_returns_flagged_sources():
    semantic_reports = {
        "jess103": {
            "grounding_flags": [
                {"file": "water_disputes.md", "overlap": 0.0}
            ]
        },
        "jess104": {
            "grounding_flags": []
        },
    }

    failures = collect_grounding_failures(semantic_reports)

    assert list(failures) == ["jess103"]
    assert len(failures["jess103"]) == 1


def test_collect_grounding_failures_returns_empty_dict_when_clean():
    semantic_reports = {
        "jess103": {"grounding_flags": []},
        "jess104": {"grounding_flags": []},
    }

    assert collect_grounding_failures(semantic_reports) == {}

def test_grounding_failures_block_promotion():
    failures = collect_required_failures(
        structure_failures={},
        provenance_failures={},
        grounding_failures={
            "jess103": [
                {"file": "water_disputes.md", "overlap": 0.0}
            ]
        },
    )

    assert "grounding" in failures
    assert failures["grounding"]["jess103"]


def test_clean_validation_allows_returns_no_required_failures():
    failures = collect_required_failures(
        structure_failures={},
        provenance_failures={},
        grounding_failures={},
    )

    assert failures == {}


def test_structure_and_provenance_failures_block_promotion():
    failures = collect_required_failures(
        structure_failures={"concept.md": ["Missing title"]},
        provenance_failures={"concept.md": ["Missing source page"]},
        grounding_failures={},
    )

    assert "structure" in failures
    assert "provenance" in failures

def test_grounding_failure_prevents_promotion():
    failures = collect_required_failures(
        structure_failures={},
        provenance_failures={},
        grounding_failures={
            "jess103": [{"file": "water_disputes.md", "overlap": 0.0}]
        },
    )

    assert can_promote_knowledge_base(failures) is False


def test_clean_validation_allows_promotion():
    failures = collect_required_failures({}, {}, {})

    assert can_promote_knowledge_base(failures) is True