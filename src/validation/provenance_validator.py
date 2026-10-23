
import json
import re
from pathlib import Path

import yaml


def load_extraction(extraction_path: str | Path) -> dict:
    """Load the cleaned extraction JSON for a source PDF."""
    extraction_path = Path(extraction_path)
    payload = json.loads(extraction_path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("Extraction JSON must contain an object.")

    if not isinstance(payload.get("pages"), list):
        raise ValueError("Extraction JSON must contain a pages list.")

    return payload


def extract_source_pages(sources: list) -> set[int]:
    """Extract page numbers from source metadata strings."""
    page_numbers = set()

    for source in sources:
        if not isinstance(source, str):
            continue

        match = re.search(
            r"\bpages?\s+(\d+(?:\s*,\s*\d+)*)",
            source,
            flags=re.IGNORECASE,
        )
        if match:
            page_numbers.update(
                int(number)
                for number in re.findall(r"\d+", match.group(1))
            )

    return page_numbers


def validate_provenance(
    okf_path: str | Path,
    extraction: dict,
) -> list[str]:
    """Check source filename and page references against an extraction."""
    okf_path = Path(okf_path)
    errors = []

    try:
        content = okf_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"Cannot read OKF file: {exc}"]

    if not content.startswith("---\n"):
        return ["Missing YAML frontmatter."]

    parts = content.split("---", 2)
    if len(parts) != 3:
        return ["Invalid frontmatter delimiters."]

    try:
        metadata = yaml.safe_load(parts[1])
    except yaml.YAMLError as exc:
        return [f"Invalid YAML: {exc}"]

    if not isinstance(metadata, dict):
        return ["Frontmatter must be a YAML mapping."]

    source_file = extraction.get("source_file")
    pages = extraction.get("pages", [])

    if not isinstance(source_file, str) or not source_file.strip():
        return ["Extraction is missing a valid source_file."]

    valid_page_numbers = {
        page.get("page_number")
        for page in pages
        if isinstance(page, dict)
        and isinstance(page.get("page_number"), int)
        and not isinstance(page.get("page_number"), bool)
    }

    sources = metadata.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("No source references in YAML metadata.")
        sources = []

    source_pages = extract_source_pages(sources)

    if not source_pages:
        errors.append("No page numbers found in YAML source metadata.")

    for source in sources:
        if isinstance(source, str) and source_file.lower() not in source.lower():
            errors.append(
                f"Source reference does not name the expected PDF "
                f"'{source_file}': {source}"
            )

    invalid_pages = source_pages - valid_page_numbers
    if invalid_pages:
        errors.append(
            "Source metadata references pages absent from the extraction: "
            + ", ".join(map(str, sorted(invalid_pages)))
        )

    body = parts[2]
    provenance_match = re.search(
        r"## Source provenance\s*(.*)",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not provenance_match:
        errors.append("Missing Source provenance section.")
        return errors

    provenance_text = provenance_match.group(1)

    pdf_match = re.search(
        r"PDF:\s*`?([^`\r\n]+?)`?\s*$",
        provenance_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if not pdf_match:
        errors.append("PDF filename missing from Source provenance section.")
    elif pdf_match.group(1).strip().lower() != source_file.lower():
        errors.append(
            f"Provenance PDF mismatch: expected '{source_file}', "
            f"found '{pdf_match.group(1).strip()}'."
        )

    pages_match = re.search(
        r"Pages:\s*([\d,\s]+)",
        provenance_text,
        flags=re.IGNORECASE,
    )

    if not pages_match:
        errors.append("Page numbers missing from Source provenance section.")
    else:
        body_pages = {
            int(number)
            for number in re.findall(r"\d+", pages_match.group(1))
        }

        if body_pages != source_pages:
            errors.append(
                "Page numbers in Markdown provenance do not match "
                "the YAML source metadata."
            )

        invalid_body_pages = body_pages - valid_page_numbers
        if invalid_body_pages:
            errors.append(
                "Markdown provenance references pages absent from "
                "the extraction: "
                + ", ".join(map(str, sorted(invalid_body_pages)))
            )

    return errors


def validate_provenance_directory(
    okf_dir: str | Path,
    extraction_path: str | Path,
) -> dict[str, list[str]]:
    """Validate provenance for every Markdown file in a directory."""
    okf_dir = Path(okf_dir)

    if not okf_dir.is_dir():
        raise NotADirectoryError(f"Directory not found: {okf_dir}")

    extraction = load_extraction(extraction_path)

    return {
    path.name: validate_provenance(path, extraction)
    for path in sorted(okf_dir.glob("*.md"))
    if path.name.lower() != "index.md"
    }