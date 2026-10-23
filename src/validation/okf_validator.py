
from pathlib import Path
import re

import yaml


REQUIRED_FIELDS = {
    "type",
    "id",
    "title",
    "description",
    "version",
    "status",
    "tags",
    "sources",
}


def validate_okf_file(file_path: str | Path) -> list[str]:
    """Validate one OKF Markdown file and return its errors."""
    file_path = Path(file_path)
    errors = []

    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"Cannot read file: {exc}"]

    if not content.startswith("---\n"):
        return ["Missing YAML frontmatter opening delimiter."]

    parts = content.split("---", 2)
    if len(parts) != 3:
        return ["Missing YAML frontmatter closing delimiter."]

    try:
        metadata = yaml.safe_load(parts[1])
    except yaml.YAMLError as exc:
        return [f"Invalid YAML frontmatter: {exc}"]

    if not isinstance(metadata, dict):
        return ["YAML frontmatter must be a mapping."]

    missing = REQUIRED_FIELDS - metadata.keys()
    if missing:
        errors.append(
            "Missing required fields: " + ", ".join(sorted(missing))
        )

    for field in ("type", "id", "title", "description", "version", "status"):
        if field in metadata and (
            not isinstance(metadata[field], str)
            or not metadata[field].strip()
        ):
            errors.append(f"'{field}' must be a non-empty string.")

    if "tags" in metadata and (
        not isinstance(metadata["tags"], list)
        or not all(isinstance(tag, str) and tag.strip()
                   for tag in metadata["tags"])
    ):
        errors.append("'tags' must be a list of non-empty strings.")

    if "sources" in metadata and not isinstance(metadata["sources"], list):
        errors.append("'sources' must be a list.")

    concept_id = metadata.get("id")
    if isinstance(concept_id, str) and not re.fullmatch(
        r"KC-[A-Z0-9_-]+-\d{3,}", concept_id
    ):
        errors.append(
            "'id' does not match the expected generated concept ID format."
        )

    markdown_body = parts[2].strip()
    if not markdown_body:
        errors.append("Markdown body is empty.")

    return errors


def validate_okf_directory(directory: str | Path) -> dict[str, list[str]]:
    """Validate all Markdown files and detect duplicate concept IDs."""
    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")

    files = sorted(directory.glob("*.md"))
    results = {}
    ids_to_files = {}

    for file_path in files:
        errors = validate_okf_file(file_path)
        results[file_path.name] = errors

        # Read metadata again only when file-level validation succeeded.
        if not errors:
            content = file_path.read_text(encoding="utf-8")
            metadata = yaml.safe_load(content.split("---", 2)[1])
            concept_id = metadata.get("id")
            if isinstance(concept_id, str):
                ids_to_files.setdefault(concept_id, []).append(file_path.name)

    for concept_id, filenames in ids_to_files.items():
        if len(filenames) > 1:
            message = f"Duplicate ID '{concept_id}' in: {', '.join(filenames)}"
            for filename in filenames:
                results[filename].append(message)

    return results
