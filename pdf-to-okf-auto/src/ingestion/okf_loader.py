"""Parse OKF concepts for the standalone conversion and QA pipeline."""
from dataclasses import dataclass
from pathlib import Path
import re

import yaml


@dataclass(frozen=True)
class EvidenceRecord:
    """A stored supporting quotation from a source PDF page."""

    page_number: int
    quote: str


@dataclass(frozen=True)
class RelatedConcept:
    """A reference to another knowledge concept."""

    title: str
    filename: str


@dataclass(frozen=True)
class OKFConceptRecord:
    """A structured OKF concept with its metadata and provenance."""

    concept_id: str
    title: str
    description: str
    tags: tuple[str, ...]
    content: str
    source_files: tuple[str, ...]
    source_pages: tuple[int, ...]
    evidence: tuple[EvidenceRecord, ...]
    related_concepts: tuple[RelatedConcept, ...]
    version: str = "1.0"
    status: str = "stable"


def _parse_frontmatter(markdown: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and Markdown body."""

    match = re.match(
        r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)",
        markdown,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError("Missing or invalid YAML frontmatter")

    metadata = yaml.safe_load(match.group(1))

    if not isinstance(metadata, dict):
        raise ValueError("YAML frontmatter must contain a mapping")

    return metadata, markdown[match.end():]


def _parse_sources(sources: list) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Extract PDF filenames and physical page numbers from source metadata."""

    source_files = []
    source_pages = set()

    for source in sources:
        if not isinstance(source, str):
            continue

        match = re.match(
            r"^\s*(.+?),\s*pages?\s+(.+?)\s*$",
            source,
            flags=re.IGNORECASE,
        )

        if not match:
            source_files.append(source.strip())
            continue

        filename = match.group(1).strip()
        page_values = match.group(2)

        if filename and filename not in source_files:
            source_files.append(filename)

        for value in page_values.split(","):
            value = value.strip()

            if value.isdigit() and int(value) > 0:
                source_pages.add(int(value))

    return tuple(source_files), tuple(sorted(source_pages))


def _extract_embedding_content(body: str) -> str:
    """Remove provenance and relationship sections from embedding text."""

    excluded_sections = {"source provenance", "related concepts"}
    content_lines = []
    skip_section = False

    for line in body.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)

        if heading:
            heading_level = len(heading.group(1))
            heading_title = heading.group(2).strip().casefold()

            if heading_title in excluded_sections:
                skip_section = True
                continue

            if skip_section and heading_level <= 2:
                skip_section = False

        if not skip_section:
            content_lines.append(line)

    content = "\n".join(content_lines).strip()

    # Remove the document title because it is already stored in metadata.
    content = re.sub(
        r"^#\s+.+?\n+",
        "",
        content,
        count=1,
    ).strip()

    return content


def _parse_evidence(items: list) -> tuple[EvidenceRecord, ...]:
    """Parse evidence quotations without modifying their text."""

    evidence = []

    for item in items:
        if not isinstance(item, dict):
            continue

        page_number = item.get("page_number")
        quote = item.get("quote")

        if (
            isinstance(page_number, int)
            and not isinstance(page_number, bool)
            and page_number > 0
            and isinstance(quote, str)
            and quote.strip()
        ):
            evidence.append(
                EvidenceRecord(
                    page_number=page_number,
                    quote=quote.strip(),
                )
            )

    return tuple(evidence)


def _parse_related_concepts(section: str) -> tuple[RelatedConcept, ...]:
    """Extract Markdown links from the Related concepts section."""

    related = []

    pattern = re.compile(
        r"^\s*-\s+\[([^\]]+)\]\(([^)]+)\)\s*$",
        flags=re.MULTILINE,
    )

    for match in pattern.finditer(section):
        related.append(
            RelatedConcept(
                title=match.group(1).strip(),
                filename=match.group(2).strip(),
            )
        )

    return tuple(related)


def _extract_related_section(body: str) -> str:
    """Extract the Related concepts section for graph metadata."""

    match = re.search(
        r"^##\s+Related concepts\s*$([\s\S]*?)"
        r"(?=^#{1,2}\s+|\Z)",
        body,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    return match.group(1).strip() if match else ""


def load_okf_concept(markdown_path: str | Path) -> OKFConceptRecord:
    """Load one OKF Markdown file into a structured concept record."""

    path = Path(markdown_path)

    if not path.is_file():
        raise FileNotFoundError(f"OKF file does not exist: {path}")

    if path.suffix.lower() != ".md":
        raise ValueError(f"Expected a Markdown file, got: {path}")

    markdown = path.read_text(encoding="utf-8")
    metadata, body = _parse_frontmatter(markdown)

    required_fields = ("id", "title", "description")

    for field in required_fields:
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise ValueError(f"Missing or invalid required metadata: {field}")

    tags = metadata.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    sources = metadata.get("sources", [])
    if not isinstance(sources, list):
        sources = []

    evidence_items = metadata.get("evidence", [])
    if not isinstance(evidence_items, list):
        evidence_items = []

    source_files, source_pages = _parse_sources(sources)
    related_section = _extract_related_section(body)

    return OKFConceptRecord(
        concept_id=metadata["id"].strip(),
        title=metadata["title"].strip(),
        description=metadata["description"].strip(),
        tags=tuple(
            tag.strip()
            for tag in tags
            if isinstance(tag, str) and tag.strip()
        ),
        content=_extract_embedding_content(body),
        source_files=source_files,
        source_pages=source_pages,
        evidence=_parse_evidence(evidence_items),
        related_concepts=_parse_related_concepts(related_section),
        version=str(metadata.get("version", "1.0")),
        status=str(metadata.get("status", "stable")),
    )


def load_okf_directory(directory: str | Path) -> list[OKFConceptRecord]:
    """Load all concept Markdown files from a directory, excluding index.md."""

    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(f"OKF directory does not exist: {directory}")

    concepts = []

    for markdown_path in sorted(directory.glob("*.md")):
        if markdown_path.name.casefold() == "index.md":
            continue

        concepts.append(load_okf_concept(markdown_path))

    return concepts
