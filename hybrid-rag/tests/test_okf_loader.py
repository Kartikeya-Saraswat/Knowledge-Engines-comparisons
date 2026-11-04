
from pathlib import Path

import pytest

from src.ingestion.okf_loader import (
    load_okf_concept,
    load_okf_directory,
)


def create_okf_file(path: Path) -> None:
    """Create a representative OKF concept."""

    content = """---
type: KnowledgeConcept
id: KC-JESS101-001
title: Resource Planning in India
description: Resource planning involves identifying and developing resources.
version: '1.0'
status: stable
tags:
- resource_planning
- india_resource_planning
sources:
- jess101.pdf, pages 3
evidence:
- page_number: 3
  quote: Resource planning is a complex process involving identification and inventory of resources across the regions.
---

# Resource Planning in India

Resource planning is a complex process involving identification and inventory
of resources across the regions.

It also involves developing appropriate technology, skills and institutions.

## Source provenance

- PDF: `jess101.pdf`
- Pages: 3

## Related concepts

- [Resource Classification](resource-classification.md)
"""

    path.write_text(content, encoding="utf-8")


def test_load_okf_concept_parses_metadata_and_provenance(tmp_path):
    path = tmp_path / "resource-planning.md"
    create_okf_file(path)

    concept = load_okf_concept(path)

    assert concept.concept_id == "KC-JESS101-001"
    assert concept.title == "Resource Planning in India"
    assert concept.description.startswith("Resource planning")
    assert concept.tags == (
        "resource_planning",
        "india_resource_planning",
    )
    assert concept.source_files == ("jess101.pdf",)
    assert concept.source_pages == (3,)
    assert len(concept.evidence) == 1
    assert concept.evidence[0].page_number == 3


def test_embedding_content_excludes_provenance_and_relationships(tmp_path):
    path = tmp_path / "resource-planning.md"
    create_okf_file(path)

    concept = load_okf_concept(path)

    assert "identification and inventory" in concept.content
    assert "Source provenance" not in concept.content
    assert "Related concepts" not in concept.content
    assert "jess101.pdf" not in concept.content
    assert "resource-classification.md" not in concept.content


def test_load_okf_concept_extracts_related_concepts(tmp_path):
    path = tmp_path / "resource-planning.md"
    create_okf_file(path)

    concept = load_okf_concept(path)

    assert len(concept.related_concepts) == 1
    assert concept.related_concepts[0].title == "Resource Classification"
    assert (
        concept.related_concepts[0].filename
        == "resource-classification.md"
    )


def test_load_okf_concept_missing_file():
    with pytest.raises(FileNotFoundError):
        load_okf_concept("missing.md")


def test_load_okf_concept_rejects_missing_frontmatter(tmp_path):
    path = tmp_path / "invalid.md"
    path.write_text("# No frontmatter", encoding="utf-8")

    with pytest.raises(ValueError, match="frontmatter"):
        load_okf_concept(path)


def test_load_okf_directory_excludes_index(tmp_path):
    concept_path = tmp_path / "resource-planning.md"
    create_okf_file(concept_path)

    (tmp_path / "index.md").write_text(
        "# Knowledge Base Index\nTotal concepts: 1",
        encoding="utf-8",
    )

    concepts = load_okf_directory(tmp_path)

    assert len(concepts) == 1
    assert concepts[0].concept_id == "KC-JESS101-001"
