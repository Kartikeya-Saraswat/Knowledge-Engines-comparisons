
import re
from itertools import combinations
from pathlib import Path

import yaml

from src.validation.provenance_validator import (
    extract_source_pages,
    load_extraction,
)


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by",
    "for", "from", "has", "have", "in", "into", "is", "it",
    "its", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "which", "with", "can", "may",
}


def tokenize(text: str) -> set[str]:
    """Return normalized words, excluding common stopwords."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 1}


def read_okf_concept(path: Path) -> dict:
    """Read a concept's metadata and knowledge content."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path.name}: missing YAML frontmatter")

    parts = text.split("---", 2)
    if len(parts) != 3:
        raise ValueError(f"{path.name}: malformed frontmatter delimiters")

    metadata = yaml.safe_load(parts[1])
    if not isinstance(metadata, dict):
        raise ValueError(f"{path.name}: frontmatter must be a mapping")

    body = re.split(
        r"^## Source provenance\s*$",
        parts[2],
        maxsplit=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )[0].strip()

    return {
        "filename": path.name,
        "id": metadata.get("id", ""),
        "title": metadata.get("title", path.stem),
        "tags": metadata.get("tags", []),
        "sources": metadata.get("sources", []),
        "body": body,
    }


def analyze_semantic_quality(
    okf_dir: str | Path,
    extraction_path: str | Path,
    low_overlap_threshold: float = 0.25,
    duplicate_threshold: float = 0.60,
) -> dict:
    """Generate advisory grounding and duplicate-concept checks."""
    okf_dir = Path(okf_dir)
    extraction = load_extraction(extraction_path)

    pages = {
        page["page_number"]: page.get("text", "")
        for page in extraction["pages"]
        if isinstance(page, dict)
        and isinstance(page.get("page_number"), int)
        and not isinstance(page.get("page_number"), bool)
    }

    concepts = []
    for path in sorted(okf_dir.glob("*.md")):
        concept = read_okf_concept(path)
        concept["tokens"] = tokenize(
            concept["title"] + " " + concept["body"]
        )

        source_pages = extract_source_pages(concept["sources"])
        source_text = " ".join(
            pages[number]
            for number in source_pages
            if number in pages
        )
        source_tokens = tokenize(source_text)

        concept_tokens = concept["tokens"]
        overlap = (
            len(concept_tokens & source_tokens) / len(concept_tokens)
            if concept_tokens else 0.0
        )

        concept["source_overlap"] = overlap
        concept["source_pages"] = sorted(source_pages)
        concepts.append(concept)

    grounding_flags = []
    for concept in concepts:
        if concept["source_overlap"] < low_overlap_threshold:
            grounding_flags.append({
                "file": concept["filename"],
                "id": concept["id"],
                "overlap": round(concept["source_overlap"], 3),
                "pages": concept["source_pages"],
                "reason": "Low exact-word overlap; manually review grounding.",
            })

    duplicate_candidates = []
    for first, second in combinations(concepts, 2):
        a = first["tokens"]
        b = second["tokens"]
        union = a | b

        similarity = len(a & b) / len(union) if union else 0.0

        if similarity >= duplicate_threshold:
            duplicate_candidates.append({
                "file_a": first["filename"],
                "file_b": second["filename"],
                "similarity": round(similarity, 3),
            })

    return {
        "concept_count": len(concepts),
        "grounding_flags": grounding_flags,
        "duplicate_candidates": duplicate_candidates,
    }
