
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.okf_validator import (
    validate_okf_file,
    validate_okf_directory,
)


def make_okf_file(path: Path, concept_id: str = "KC-TEST-001") -> Path:
    """Create a minimal valid OKF Markdown file for testing."""
    content = f"""---
type: KnowledgeConcept
id: {concept_id}
title: Test Concept
description: A test knowledge concept.
version: "1.0"
status: stable
tags:
  - testing
sources:
  - sample.pdf, pages 1
---

# Test Concept

This is test content.
"""
    path.write_text(content, encoding="utf-8")
    return path


def test_valid_okf_file_passes(tmp_path):
    file_path = make_okf_file(tmp_path / "valid.md")

    assert validate_okf_file(file_path) == []


def test_missing_required_field_fails(tmp_path):
    file_path = tmp_path / "missing_field.md"
    file_path.write_text(
        """---
type: KnowledgeConcept
id: KC-TEST-001
title: Test Concept
version: "1.0"
status: stable
tags: []
sources: []
---

# Test Concept

Content.
""",
        encoding="utf-8",
    )

    errors = validate_okf_file(file_path)

    assert any("description" in error for error in errors)


def test_malformed_yaml_fails(tmp_path):
    file_path = tmp_path / "malformed.md"
    file_path.write_text(
        """---
type: [invalid
id: KC-TEST-001
---

# Test Concept
""",
        encoding="utf-8",
    )

    errors = validate_okf_file(file_path)

    assert any("Invalid YAML frontmatter" in error for error in errors)


def test_empty_markdown_body_fails(tmp_path):
    file_path = tmp_path / "empty_body.md"
    file_path.write_text(
        """---
type: KnowledgeConcept
id: KC-TEST-001
title: Test Concept
description: A test concept.
version: "1.0"
status: stable
tags: []
sources: []
---
""",
        encoding="utf-8",
    )

    errors = validate_okf_file(file_path)

    assert any("Markdown body is empty" in error for error in errors)


def test_duplicate_ids_fail_directory_validation(tmp_path):
    make_okf_file(tmp_path / "concept_a.md", "KC-TEST-001")
    make_okf_file(tmp_path / "concept_b.md", "KC-TEST-001")

    results = validate_okf_directory(tmp_path)

    assert any("Duplicate ID" in error for error in results["concept_a.md"])
    assert any("Duplicate ID" in error for error in results["concept_b.md"])
