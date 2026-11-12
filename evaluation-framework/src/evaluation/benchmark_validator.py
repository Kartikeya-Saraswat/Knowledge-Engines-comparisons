"""Validate audited evidence annotations without generating answers or labels."""

import hashlib
import re
from pathlib import Path

import yaml

from src.evaluation.text_matching import contains_required_quotes, normalize_excerpt


def load_concept_texts(directory):
    """Read text actually available to retrieval; exclude provenance and links."""
    concepts = {}
    for path in sorted(Path(directory).glob("*.md")):
        if path.name.casefold() == "index.md":
            continue
        content = path.read_text(encoding="utf-8")
        match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", content, re.S)
        if not match:
            raise ValueError(f"{path.name}: missing YAML frontmatter")
        metadata = yaml.safe_load(match.group(1))
        if not isinstance(metadata, dict) or not isinstance(metadata.get("id"), str):
            raise ValueError(f"{path.name}: missing concept ID")
        concept_id = metadata["id"]
        if concept_id in concepts:
            raise ValueError(f"Duplicate concept ID: {concept_id}")
        lines = []
        skip = False
        for line in content[match.end():].splitlines():
            heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if heading:
                if heading.group(2).casefold() in {"source provenance", "related concepts"}:
                    skip = True
                    continue
                if len(heading.group(1)) <= 2:
                    skip = False
            if not skip:
                lines.append(line)
        concepts[concept_id] = "\n".join([
            str(metadata.get("title", "")), str(metadata.get("description", "")),
            "Tags: " + ", ".join(str(tag) for tag in metadata.get("tags", [])),
            *lines,
        ])
    if not concepts:
        raise ValueError(f"No OKF concepts found in {directory}")
    return concepts


def validate_benchmark(questions, concept_texts, pdf_pages=None):
    """Check structure, label consistency, and supporting excerpt locations.

    This does not establish semantic entailment. Full and partial labels remain
    editorial judgments whose selected excerpts are checked against frozen text.
    Joint anchors are intentionally disallowed: split jointly required facts.
    """
    errors = []
    if not isinstance(questions, list) or not questions:
        return ["Benchmark must be a nonempty question list"]
    seen_questions = set()
    for question in questions:
        if not isinstance(question, dict):
            errors.append("Question must be an object")
            continue
        qid = question.get("id")
        if not isinstance(qid, str) or not qid:
            errors.append("Question ID must be a nonempty string")
            continue
        if qid in seen_questions:
            errors.append(f"Invalid or duplicate question ID: {qid}")
        seen_questions.add(qid)
        if not isinstance(question.get("answerable"), bool):
            errors.append(f"{qid}: answerable must be boolean")
        units = question.get("evidence_units", [])
        if not isinstance(units, list):
            errors.append(f"{qid}: evidence_units must be a list")
            continue
        if bool(units) != question.get("answerable"):
            errors.append(f"{qid}: answerable questions need evidence; unanswerable questions must have none")
        seen_units, seen_facts = set(), set()
        for unit in units:
            if not isinstance(unit, dict):
                errors.append(f"{qid}: evidence unit must be an object")
                continue
            label = f"{qid}/{unit.get('id')}"
            uid = unit.get("id")
            if not isinstance(uid, str) or not uid:
                errors.append(f"{label}: evidence ID must be a nonempty string")
                continue
            if uid in seen_units:
                errors.append(f"{label}: invalid or duplicate evidence ID")
            seen_units.add(uid)
            fact = unit.get("fact", "")
            if not isinstance(fact, str) or not fact.strip():
                errors.append(f"{label}: missing explicit fact")
            elif normalize_excerpt(fact) in seen_facts:
                errors.append(f"{label}: duplicate fact")
            else:
                seen_facts.add(normalize_excerpt(fact))
            if type(unit.get("relevance")) is not int or unit["relevance"] not in (1, 2, 3):
                errors.append(f"{label}: relevance must be an integer from 1 to 3")
            if unit.get("pdf_anchor_mode") != "any":
                errors.append(f"{label}: anchors must be independently sufficient alternatives; split joint facts")
            anchors = unit.get("pdf_anchors", [])
            if not isinstance(anchors, list) or not anchors:
                errors.append(f"{label}: missing PDF anchors")
                anchors = []
            for anchor in anchors:
                if not isinstance(anchor, dict):
                    errors.append(f"{label}: PDF anchor must be an object")
                    continue
                document, pages, quote = anchor.get("document"), anchor.get("pages"), anchor.get("quote")
                if not isinstance(document, str) or not document:
                    errors.append(f"{label}: invalid PDF document")
                    continue
                if not isinstance(pages, list) or not pages or any(type(p) is not int or p < 1 for p in pages):
                    errors.append(f"{label}: invalid physical PDF pages")
                    continue
                if not isinstance(quote, str) or not normalize_excerpt(quote):
                    errors.append(f"{label}: missing source quote")
                    continue
                if "required_quotes" in anchor and not contains_required_quotes(
                    quote, anchor["required_quotes"],
                ):
                    errors.append(f"{label}: required PDF excerpts must occur completely and in order in the source quote")
                if pdf_pages is not None:
                    available = pdf_pages.get(document)
                    if available is None or any(p > len(available) for p in pages):
                        errors.append(f"{label}: unknown document or out-of-range PDF page")
                    else:
                        cited = "\n".join(available[p - 1] for p in pages)
                        if not contains_required_quotes(cited, [quote]):
                            errors.append(f"{label}: PDF quote not found on cited pages")

            ids = unit.get("okf_concept_ids")
            if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
                errors.append(f"{label}: okf_concept_ids must be a list of strings")
                ids = []
            if len(ids) != len(set(ids)):
                errors.append(f"{label}: duplicate full-support concept IDs")
            supports = unit.get("okf_support")
            if not isinstance(supports, list):
                errors.append(f"{label}: missing OKF support audit")
                supports = []
            full, seen_concepts = set(), set()
            for support in supports:
                if not isinstance(support, dict):
                    errors.append(f"{label}: OKF support annotation must be an object")
                    continue
                concept_id = support.get("concept_id")
                kind = support.get("support")
                if not isinstance(concept_id, str):
                    errors.append(f"{label}: support concept ID must be a string")
                    continue
                if concept_id in seen_concepts:
                    errors.append(f"{label}: duplicate support annotation for {concept_id}")
                seen_concepts.add(concept_id)
                if kind not in ("full", "partial"):
                    errors.append(f"{label}: support must be full or partial")
                if kind == "full":
                    full.add(concept_id)
                reason = support.get("reason")
                if kind == "partial" and (not isinstance(reason, str) or not reason.strip()):
                    errors.append(f"{label}: partial support needs a reason")
                quotes = support.get("quotes")
                if not isinstance(quotes, list) or not quotes:
                    errors.append(f"{label}: support annotation needs excerpts")
                    quotes = []
                if concept_id not in concept_texts:
                    errors.append(f"{label}: unknown OKF concept {concept_id}")
                else:
                    for quote in quotes:
                        if not contains_required_quotes(concept_texts[concept_id], [quote]):
                            errors.append(f"{label}: supporting quote not found in {concept_id}")
            if set(ids) != full:
                errors.append(f"{label}: full-support IDs must exactly match full annotations; partial support earns no full credit")
            note = unit.get("coverage_note")
            if not ids and (not isinstance(note, str) or not note.strip()):
                errors.append(f"{label}: missing coverage explanation for an unmapped fact")
    return errors


def validate_snapshot(snapshot, benchmark_path, concept_directory, pdf_directory):
    """Detect a changed benchmark, corpus, or source bundle after the audit."""
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    errors = []
    if snapshot.get("schema_version") != 2:
        errors.append("Unsupported evidence audit version")
    if digest(Path(benchmark_path)) != snapshot.get("benchmark_sha256"):
        errors.append("Benchmark differs from the audited revision")
    for field, directory, pattern in [
        ("concept_files", Path(concept_directory), "*.md"),
        ("pdf_files", Path(pdf_directory), "*.pdf"),
    ]:
        expected = {item["filename"]: item["sha256"] for item in snapshot.get(field, [])}
        actual = {p.name: digest(p) for p in directory.glob(pattern) if p.name.casefold() != "index.md"}
        if actual != expected:
            errors.append(f"{field}: files differ from the frozen audit inputs")
    return errors
