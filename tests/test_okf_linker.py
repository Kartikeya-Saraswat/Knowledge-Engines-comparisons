
from pathlib import Path

import yaml
from unittest.mock import MagicMock, patch

from src.generation.okf_linker import write_related_links
from src.generation.okf_linker import write_concept_index
from src.generation.okf_linker import find_related_concepts
from src.generation.okf_linker import load_concept_catalogue


def test_load_concept_catalogue_reads_metadata(tmp_path: Path):
    metadata = {
        "type": "KnowledgeConcept",
        "id": "KC-TEST-001",
        "title": "Resource Planning",
        "description": "An overview of resource planning.",
        "tags": ["resources", "planning"],
    }

    frontmatter = yaml.safe_dump(
        metadata,
        sort_keys=False,
        allow_unicode=True,
    ).strip()

    (tmp_path / "resource-planning.md").write_text(
        f"---\n{frontmatter}\n---\n\n"
        "# Resource Planning\n\n"
        "Resource planning involves identifying resources.\n",
        encoding="utf-8",
    )

    catalogue = load_concept_catalogue(tmp_path)

    assert len(catalogue) == 1
    assert catalogue[0]["id"] == "KC-TEST-001"
    assert catalogue[0]["title"] == "Resource Planning"
    assert catalogue[0]["filename"] == "resource-planning.md"
    assert "identifying resources" in catalogue[0]["content"]


def test_load_concept_catalogue_rejects_duplicate_ids(
    tmp_path: Path,
):
    for filename, title in [
        ("first.md", "First concept"),
        ("second.md", "Second concept"),
    ]:
        metadata = {
            "id": "KC-TEST-001",
            "title": title,
        }
        frontmatter = yaml.safe_dump(metadata).strip()

        (tmp_path / filename).write_text(
            f"---\n{frontmatter}\n---\n\n# {title}\n",
            encoding="utf-8",
        )

    try:
        load_concept_catalogue(tmp_path)
    except ValueError as exc:
        assert "Duplicate concept IDs" in str(exc)
    else:
        raise AssertionError("Expected duplicate IDs to be rejected.")


def test_find_related_concepts_filters_invalid_and_self_links():
    catalogue = [
        {"id": "KC-TEST-001", "title": "Resources"},
        {"id": "KC-TEST-002", "title": "Resource Planning"},
    ]

    mock_response = MagicMock()
    mock_response.choices[0].message.content = """
    {
      "relationships": [
        {
          "source_id": "KC-TEST-001",
          "related_ids": [
            "KC-TEST-001",
            "KC-TEST-002",
            "KC-TEST-002",
            "KC-UNKNOWN-999"
          ]
        }
      ]
    }
    """

    with patch("src.generation.okf_linker.Groq") as mock_groq:
        mock_groq.return_value.chat.completions.create.return_value = (
            mock_response
        )

        with patch.dict(
            "os.environ",
            {"GROQ_API_KEY": "test-key"},
        ):
            result = find_related_concepts(catalogue)

    assert result["KC-TEST-001"] == ["KC-TEST-002"]
    assert result["KC-TEST-002"] == []


def test_find_related_concepts_respects_max_related():
    catalogue = [
        {"id": "KC-TEST-001", "title": "Resources"},
        {"id": "KC-TEST-002", "title": "Planning"},
        {"id": "KC-TEST-003", "title": "Conservation"},
    ]

    mock_response = MagicMock()
    mock_response.choices[0].message.content = """
    {
      "relationships": [
        {
          "source_id": "KC-TEST-001",
          "related_ids": [
            "KC-TEST-002",
            "KC-TEST-003"
          ]
        }
      ]
    }
    """

    with patch("src.generation.okf_linker.Groq") as mock_groq:
        mock_groq.return_value.chat.completions.create.return_value = (
            mock_response
        )

        with patch.dict(
            "os.environ",
            {"GROQ_API_KEY": "test-key"},
        ):
            result = find_related_concepts(
                catalogue,
                max_related=1,
            )

    assert result["KC-TEST-001"] == ["KC-TEST-002"]
    assert len(result["KC-TEST-001"]) <= 1


def test_find_related_concepts_zero_limit_skips_api_call():
    catalogue = [
        {"id": "KC-TEST-001", "title": "Resources"},
    ]

    with patch("src.generation.okf_linker.Groq") as mock_groq:
        result = find_related_concepts(catalogue, max_related=0)

    assert result == {"KC-TEST-001": []}
    mock_groq.assert_not_called()

def test_write_related_links_creates_relative_links(tmp_path: Path):
    first = tmp_path / "resources.md"
    second = tmp_path / "planning.md"

    first.write_text(
        "---\nid: KC-TEST-001\ntitle: Resources\n---\n\n"
        "# Resources\n\nResource information.\n",
        encoding="utf-8",
    )
    second.write_text(
        "---\nid: KC-TEST-002\ntitle: Planning\n---\n\n"
        "# Planning\n\nPlanning information.\n",
        encoding="utf-8",
    )

    catalogue = [
        {
            "id": "KC-TEST-001",
            "title": "Resources",
            "filename": first.name,
            "path": str(first),
        },
        {
            "id": "KC-TEST-002",
            "title": "Planning",
            "filename": second.name,
            "path": str(second),
        },
    ]

    write_related_links(
        catalogue,
        {"KC-TEST-001": ["KC-TEST-002"], "KC-TEST-002": []},
    )

    updated = first.read_text(encoding="utf-8")
    assert "## Related concepts" in updated
    assert "[Planning](planning.md)" in updated


def test_write_related_links_is_idempotent(tmp_path: Path):
    first = tmp_path / "resources.md"
    second = tmp_path / "planning.md"

    first.write_text(
        "# Resources\n",
        encoding="utf-8",
    )
    second.write_text(
        "# Planning\n",
        encoding="utf-8",
    )

    catalogue = [
        {
            "id": "KC-TEST-001",
            "title": "Resources",
            "filename": first.name,
            "path": str(first),
        },
        {
            "id": "KC-TEST-002",
            "title": "Planning",
            "filename": second.name,
            "path": str(second),
        },
    ]
    relationships = {"KC-TEST-001": ["KC-TEST-002"]}

    write_related_links(catalogue, relationships)
    write_related_links(catalogue, relationships)

    updated = first.read_text(encoding="utf-8")
    assert updated.count("## Related concepts") == 1
    assert updated.count("[Planning](planning.md)") == 1


def test_write_concept_index_creates_links_and_is_repeatable(
    tmp_path: Path,
):
    catalogue = [
        {
            "id": "KC-TEST-002",
            "title": "Resource Planning",
            "filename": "resource-planning.md",
            "description": "The stages of resource planning.",
            "tags": ["resources", "planning"],
        },
        {
            "id": "KC-TEST-001",
            "title": "Resources",
            "filename": "resources.md",
            "description": "An overview of resources.",
            "tags": ["resources"],
        },
    ]

    index_path = write_concept_index(catalogue, tmp_path)
    first_content = index_path.read_text(encoding="utf-8")

    write_concept_index(catalogue, tmp_path)
    second_content = index_path.read_text(encoding="utf-8")

    assert index_path.name == "index.md"
    assert "Total concepts: 2" in first_content
    assert "[Resources](resources.md)" in first_content
    assert "[Resource Planning](resource-planning.md)" in first_content
    assert "**Tags:** resources, planning" in first_content
    assert first_content == second_content
