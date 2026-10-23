
from pathlib import Path

from src.validation.link_validator import validate_markdown_links


def test_valid_internal_link_passes(tmp_path: Path):
    (tmp_path / "target.md").write_text(
        "# Target\n",
        encoding="utf-8",
    )
    (tmp_path / "source.md").write_text(
        "[Target](target.md)\n",
        encoding="utf-8",
    )

    assert validate_markdown_links(tmp_path) == []


def test_broken_internal_link_is_reported(tmp_path: Path):
    (tmp_path / "source.md").write_text(
        "[Missing](missing.md)\n",
        encoding="utf-8",
    )

    errors = validate_markdown_links(tmp_path)

    assert len(errors) == 1
    assert "source.md" in errors[0]
    assert "missing.md" in errors[0]


def test_external_links_are_ignored(tmp_path: Path):
    (tmp_path / "source.md").write_text(
        "[Website](https://example.com/page)\n"
        "[Anchor](#section)\n",
        encoding="utf-8",
    )

    assert validate_markdown_links(tmp_path) == []
