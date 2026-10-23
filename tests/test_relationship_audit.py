
from pathlib import Path

from src.validation.relationship_audit import audit_relationships


def test_reports_missing_reverse_link(tmp_path: Path):
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"

    first.write_text(
        "## Related concepts\n\n- [Second](second.md)\n",
        encoding="utf-8",
    )
    second.write_text("# Second\n", encoding="utf-8")

    catalogue = [
        {
            "id": "KC-001",
            "filename": first.name,
            "path": str(first),
        },
        {
            "id": "KC-002",
            "filename": second.name,
            "path": str(second),
        },
    ]

    findings = audit_relationships(catalogue)

    assert findings == [
        {
            "source_id": "KC-001",
            "target_id": "KC-002",
            "issue": "missing_reverse_link",
        }
    ]


def test_does_not_flag_reciprocal_links(tmp_path: Path):
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"

    first.write_text(
        "## Related concepts\n\n- [Second](second.md)\n",
        encoding="utf-8",
    )
    second.write_text(
        "## Related concepts\n\n- [First](first.md)\n",
        encoding="utf-8",
    )

    catalogue = [
        {
            "id": "KC-001",
            "filename": first.name,
            "path": str(first),
        },
        {
            "id": "KC-002",
            "filename": second.name,
            "path": str(second),
        },
    ]

    assert audit_relationships(catalogue) == []


def test_handles_concept_without_related_section(tmp_path: Path):
    concept = tmp_path / "standalone.md"
    concept.write_text("# Standalone\n", encoding="utf-8")

    catalogue = [{
        "id": "KC-001",
        "filename": concept.name,
        "path": str(concept),
    }]

    assert audit_relationships(catalogue) == []


def test_ignores_links_outside_catalogue(tmp_path: Path):
    concept = tmp_path / "first.md"
    concept.write_text(
        "## Related concepts\n\n- [External](external.md)\n",
        encoding="utf-8",
    )

    catalogue = [{
        "id": "KC-001",
        "filename": concept.name,
        "path": str(concept),
    }]

    assert audit_relationships(catalogue) == []


def test_reports_each_one_way_link_once(tmp_path: Path):
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"

    first.write_text(
        "## Related concepts\n\n- [Second](second.md)\n",
        encoding="utf-8",
    )
    second.write_text("# Second\n", encoding="utf-8")

    catalogue = [
        {"id": "KC-001", "filename": first.name, "path": str(first)},
        {"id": "KC-002", "filename": second.name, "path": str(second)},
    ]

    findings = audit_relationships(catalogue)

    assert len(findings) == 1
    assert findings[0]["source_id"] == "KC-001"
    assert findings[0]["target_id"] == "KC-002"

