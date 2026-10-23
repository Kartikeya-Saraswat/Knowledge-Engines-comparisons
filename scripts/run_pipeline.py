from pypdf import PdfReader
import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_extractor import extract_pdf_pages
from src.ingestion.save_extraction import save_extracted_pages
from src.ingestion.clean_extractions import clean_extraction_file
from src.generation.okf_generator import (
    generate_concepts,
    write_okf_concepts,
)
from src.generation.okf_linker import (
    find_related_concepts,
    load_concept_catalogue,
    write_concept_index,
    write_related_links,
)
from src.validation.okf_validator import validate_okf_directory
from src.validation.provenance_validator import (
    validate_provenance_directory,
)
from src.validation.semantic_quality import analyze_semantic_quality
from src.validation.text_quality import inspect_file
from src.validation.link_validator import validate_markdown_links
from src.validation.relationship_audit import audit_relationships


BATCH_SIZE = 3

def build_page_plan(pdf_files, max_pages=None, max_batches=None):
    """Plan generation pages without calling an LLM."""
    page_limits = {}
    pages_remaining = max_pages
    batches_remaining = max_batches

    for pdf_path in pdf_files:
        page_count = len(PdfReader(str(pdf_path)).pages)
        allowed = page_count

        if pages_remaining is not None:
            allowed = min(allowed, pages_remaining)

        if batches_remaining is not None:
            allowed = min(allowed, batches_remaining * BATCH_SIZE)

        page_limits[pdf_path.stem] = allowed

        batches_used = (
            (allowed + BATCH_SIZE - 1) // BATCH_SIZE
        )

        if pages_remaining is not None:
            pages_remaining -= allowed

        if batches_remaining is not None:
            batches_remaining -= batches_used

    return page_limits

def resolve_path(value: str) -> Path:
    """Resolve a relative path from the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process a folder of PDFs into a unified OKF knowledge base."
    )
    parser.add_argument(
        "--input-dir",
        default="data/input/sample",
        help="Folder containing PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output/batch_pipeline",
        help="Dedicated output directory for this pipeline.",
    )
    parser.add_argument(
        "--max-related",
        type=int,
        default=3,
        help="Maximum related concepts per concept.",
    )

    parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Estimate workload without making API calls or writing outputs.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum total pages to send for concept generation.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum total concept-generation API calls.",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Enable LLM-based cross-chapter relationship discovery.",
    )

    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir)
    output_root = resolve_path(args.output_dir)

    if args.max_related < 0:
        parser.error("--max-related must be zero or greater.")

    if args.max_pages is not None and args.max_pages < 1:
        parser.error("--max-pages must be at least 1.")

    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be at least 1.")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    pdf_files = sorted(input_dir.rglob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {input_dir}")

    # Output filenames and concept IDs use each PDF's stem.
    stems = [path.stem for path in pdf_files]
    if len(stems) != len(set(stems)):
        raise ValueError(
            "Multiple PDFs have the same filename stem. "
            "Rename them to unique names before processing."
        )

    page_limits = build_page_plan(
        pdf_files,
        max_pages=args.max_pages,
        max_batches=args.max_batches,
    )

    planned_pages = sum(page_limits.values())
    planned_batches = sum(
        (count + BATCH_SIZE - 1) // BATCH_SIZE
        for count in page_limits.values()
    )

    print("\nWorkload estimate")
    print(f"PDFs discovered: {len(pdf_files)}")
    print(f"Pages selected for generation: {planned_pages}")
    print(f"Generation batches/API calls: {planned_batches}")
    print(f"Batch size: {BATCH_SIZE} pages")
    print(f"Cross-chapter linking enabled: {args.link}")

    for pdf_path in pdf_files:
        print(
            f"  {pdf_path.name}: "
            f"{page_limits[pdf_path.stem]} pages selected"
        )

    if args.dry_run:
        print("\nDry run complete. No Groq calls were made.")
        print("No output files were written.")
        return 0

    extracted_dir = output_root / "extracted"
    cleaned_dir = output_root / "cleaned"
    chapters_dir = output_root / "chapters"
    unified_dir = output_root / "unified"
    reports_dir = output_root / "reports"

    for directory in (
        extracted_dir,
        cleaned_dir,
        chapters_dir,
        reports_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # Rebuild the unified bundle so removed concepts cannot remain stale.
    if unified_dir.exists():
        shutil.rmtree(unified_dir)
    unified_dir.mkdir(parents=True)

    print(f"Input folder: {input_dir}")
    print(f"PDFs discovered: {len(pdf_files)}")

    extraction_paths = {}

    # Stage 1: extract and clean every PDF.
    for pdf_path in pdf_files:
        print(f"\n[1/5] Extracting {pdf_path.name}...")
        pages = extract_pdf_pages(pdf_path)

        extracted_path = save_extracted_pages(
            pdf_path, pages, extracted_dir
        )
        cleaned_path = clean_extraction_file(
            extracted_path, cleaned_dir
        )

        extraction_paths[pdf_path.stem] = cleaned_path
        print(f"  Pages extracted: {len(pages)}")
        print(f"  Cleaned JSON: {cleaned_path.name}")

    # Stage 2: generate a separate concept bundle per chapter.
    for pdf_path in pdf_files:
        source_name = pdf_path.stem
        cleaned_path = extraction_paths[source_name]
        payload = json.loads(
            cleaned_path.read_text(encoding="utf-8")
        )
        chapter_dir = chapters_dir / source_name

        if chapter_dir.exists():
            shutil.rmtree(chapter_dir)
        chapter_dir.mkdir(parents=True)

        total_written = 0
        pages = payload["pages"][:page_limits[source_name]]

        print(f"\n[2/5] Generating concepts for {source_name}...")

        for start in range(0, len(pages), BATCH_SIZE):
            batch = pages[start:start + BATCH_SIZE]
            concepts = generate_concepts(
                batch, payload["source_file"]
            )

            paths = write_okf_concepts(
                concepts,
                payload["source_file"],
                chapter_dir,
                start_index=total_written + 1,
            )
            total_written += len(paths)

        print(f"  Concepts generated: {total_written}")

        if total_written == 0:
            print(f"  WARNING: No concepts generated for {source_name}.")

    # Stage 3: combine concepts, using unique filenames.
    print("\n[3/5] Building unified knowledge base...")

    copied_count = 0
    for pdf_path in pdf_files:
        chapter_dir = chapters_dir / pdf_path.stem

        for concept_path in sorted(chapter_dir.glob("*.md")):
            if concept_path.name.lower() == "index.md":
                continue

            target_name = f"{pdf_path.stem}__{concept_path.name}"
            shutil.copy2(
                concept_path,
                unified_dir / target_name,
            )
            copied_count += 1

    if copied_count == 0:
        raise RuntimeError("No concepts were generated across the PDFs.")

    catalogue = load_concept_catalogue(unified_dir)
    if not catalogue:
        raise RuntimeError("Could not load the unified concept catalogue.")

    print(f"  Unified concepts: {len(catalogue)}")
    
    if args.link:
        print("  Discovering relationships across chapters...")

        relationships = find_related_concepts(
            catalogue,
            max_related=args.max_related,
        )
        write_related_links(catalogue, relationships)
    else:
        print("  Skipping LLM-based relationship discovery.")

    index_path = write_concept_index(catalogue, unified_dir)

    structure_results = validate_okf_directory(unified_dir)
    structure_failures = {
        name: errors
        for name, errors in structure_results.items()
        if errors
    }

    provenance_failures = {}
    semantic_reports = {}

    for pdf_path in pdf_files:
        source_name = pdf_path.stem
        chapter_dir = chapters_dir / source_name
        extraction_path = extraction_paths[source_name]

        results = validate_provenance_directory(
            chapter_dir, extraction_path
        )
        for filename, errors in results.items():
            if errors:
                provenance_failures[
                    f"{source_name}/{filename}"
                ] = errors

        semantic_reports[source_name] = analyze_semantic_quality(
            chapter_dir, extraction_path
        )

    text_findings = []
    for path in sorted(unified_dir.glob("*.md")):
        if path.name.lower() != "index.md":
            text_findings.extend(inspect_file(path))

    link_errors = validate_markdown_links(unified_dir)
    relationship_findings = audit_relationships(catalogue)

    # Semantic overlap and missing reverse links are advisory.
    summary = {
        "input_dir": str(input_dir),
        "pdf_count": len(pdf_files),
        "concept_count": len(catalogue),
        "structure_failures": structure_failures,
        "provenance_failures": provenance_failures,
        "text_findings": text_findings,
        "broken_links": link_errors,
        "semantic_reports": semantic_reports,
        "missing_reverse_links": relationship_findings,
        "index": str(index_path),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "pipeline_summary.json"
    report_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Stage 5: print a concise final report.
    print("\n[5/5] Pipeline summary")
    print(f"  PDFs processed: {len(pdf_files)}")
    print(f"  Concepts created: {len(catalogue)}")
    print(f"  Structure failures: {len(structure_failures)}")
    print(f"  Provenance failures: {len(provenance_failures)}")
    print(f"  Text-quality findings: {len(text_findings)}")
    print(f"  Broken links: {len(link_errors)}")
    print(f"  Missing reverse links (advisory): {len(relationship_findings)}")
    print(f"  Unified knowledge base: {unified_dir}")
    print(f"  Index: {index_path}")
    print(f"  Full report: {report_path}")

    failures = (
        structure_failures
        or provenance_failures
        or text_findings
        or link_errors
    )

    if failures:
        print("\nPipeline completed with validation failures.")
        return 1

    print("\nPipeline completed; required checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
