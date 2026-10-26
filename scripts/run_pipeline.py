
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from pypdf import PdfReader

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
    load_extraction,
    validate_provenance,
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

        batches_used = (allowed + BATCH_SIZE - 1) // BATCH_SIZE

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
        description="Process PDFs into a unified OKF knowledge base."
    )
    parser.add_argument(
        "--input-dir",
        default="data/input/sample",
        help="Folder containing PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output/lead_demo",
        help="Output directory for the final knowledge base and report.",
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
        help="Estimate workload without calling the model or writing outputs.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum total pages selected for concept generation.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum total concept-generation batches.",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Enable LLM-based relationship discovery.",
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
    print(f"Generation batches: {planned_batches}")
    print(f"Batch size: {BATCH_SIZE} pages")
    print(f"Cross-chapter linking enabled: {args.link}")

    for pdf_path in pdf_files:
        print(
            f"  {pdf_path.name}: "
            f"{page_limits[pdf_path.stem]} pages selected"
        )

    if args.dry_run:
        print("\nDry run complete. No model calls were made.")
        print("No output files were written.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    reports_dir = output_root / "reports"
    unified_dir = output_root / "unified"
    staging_dir = output_root / ".unified_staging"

    # Keep the previous unified knowledge base intact until the new
    # knowledge base passes its required validation checks.
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    extraction_paths = {}
    generated_by_source = {pdf.stem: [] for pdf in pdf_files}

    try:
        # Intermediates live outside the permanent output directory.
        with tempfile.TemporaryDirectory(
            prefix="pdf_to_okf_"
        ) as temporary_root:
            temporary_root = Path(temporary_root)
            extracted_dir = temporary_root / "extracted"
            cleaned_dir = temporary_root / "cleaned"
            generated_dir = temporary_root / "generated"
            validation_dir = temporary_root / "validation"

            for directory in (
                extracted_dir,
                cleaned_dir,
                generated_dir,
                validation_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)

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

            # Stage 2: generate concepts per PDF in temporary storage.
            # Only PDFs with selected pages receive a temporary folder.
            for pdf_path in pdf_files:
                source_name = pdf_path.stem
                cleaned_path = extraction_paths[source_name]
                payload = json.loads(
                    cleaned_path.read_text(encoding="utf-8")
                )
                pages = payload["pages"][:page_limits[source_name]]

                if not pages:
                    print(
                        f"\n[2/5] Skipping {source_name}: "
                        "no pages selected."
                    )
                    continue

                temporary_source_dir = generated_dir / source_name
                temporary_source_dir.mkdir(parents=True, exist_ok=True)
                total_written = 0

                print(
                    f"\n[2/5] Generating concepts for {source_name}..."
                )

                for start in range(0, len(pages), BATCH_SIZE):
                    batch = pages[start:start + BATCH_SIZE]
                    concepts = generate_concepts(
                        batch, payload["source_file"]
                    )

                    paths = write_okf_concepts(
                        concepts,
                        payload["source_file"],
                        temporary_source_dir,
                        start_index=total_written + 1,
                    )
                    total_written += len(paths)

                print(f"  Concepts generated: {total_written}")

                if total_written == 0:
                    print(
                        f"  WARNING: No concepts generated for {source_name}."
                    )

                # Stage 3: move source-prefixed concepts into staging.
                # Temporary per-PDF files are not retained in final output.
                for concept_path in sorted(
                    temporary_source_dir.glob("*.md")
                ):
                    if concept_path.name.lower() == "index.md":
                        continue

                    target = staging_dir / (
                        f"{source_name}__{concept_path.name}"
                    )
                    shutil.copy2(concept_path, target)
                    generated_by_source[source_name].append(target)

            print("\n[3/5] Building unified knowledge base...")

            concept_count = sum(
                len(paths) for paths in generated_by_source.values()
            )
            if concept_count == 0:
                raise RuntimeError(
                    "No concepts were generated across the PDFs."
                )

            catalogue = load_concept_catalogue(staging_dir)
            if not catalogue:
                raise RuntimeError(
                    "Could not load the unified concept catalogue."
                )

            print(f"  Unified concepts: {len(catalogue)}")

            if args.link:
                print("  Discovering relationships...")
                relationships = find_related_concepts(
                    catalogue,
                    max_related=args.max_related,
                )
                write_related_links(catalogue, relationships)
            else:
                print("  Skipping LLM-based relationship discovery.")

            index_path = write_concept_index(catalogue, staging_dir)

            # Stage 4: validate the staged knowledge base.
            structure_results = validate_okf_directory(staging_dir)
            structure_failures = {
                name: errors
                for name, errors in structure_results.items()
                if errors
            }

            provenance_failures = {}
            semantic_reports = {}

            for pdf_path in pdf_files:
                source_name = pdf_path.stem
                extraction_path = extraction_paths[source_name]
                extraction = load_extraction(extraction_path)
                source_concepts = generated_by_source[source_name]

                for concept_path in source_concepts:
                    errors = validate_provenance(concept_path, extraction)
                    if errors:
                        provenance_failures[
                            concept_path.name
                        ] = errors

                # The semantic analyzer accepts a directory. Use a
                # temporary source-specific directory so each report
                # only compares concepts against their own PDF.
                source_validation_dir = (
                    validation_dir / source_name
                )
                source_validation_dir.mkdir(
                    parents=True, exist_ok=True
                )

                for concept_path in source_concepts:
                    shutil.copy2(
                        concept_path,
                        source_validation_dir / concept_path.name,
                    )

                semantic_reports[source_name] = analyze_semantic_quality(
                    source_validation_dir,
                    extraction_path,
                )

            text_findings = []
            for path in sorted(staging_dir.glob("*.md")):
                if path.name.lower() != "index.md":
                    text_findings.extend(inspect_file(path))

            link_errors = validate_markdown_links(staging_dir)
            relationship_findings = audit_relationships(catalogue)

        # Temporary extraction, cleaning and validation files are now
        # removed automatically when the TemporaryDirectory closes.

        failures = (
            structure_failures
            or provenance_failures
            or text_findings
            or link_errors
        )

        # Promote only a knowledge base that passes required checks.
        if not failures:
            if unified_dir.exists():
                shutil.rmtree(unified_dir)
            staging_dir.rename(unified_dir)
            final_index_path = unified_dir / "index.md"
            print("\nNew unified knowledge base passed required checks.")
        else:
            final_index_path = staging_dir / "index.md"
            print(
                "\nRequired checks failed. The previous unified knowledge "
                "base was preserved."
            )
            print(f"Failed candidate retained at: {staging_dir}")

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
            "index": str(final_index_path),
            "required_checks_passed": not bool(failures),
        }

        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "pipeline_summary.json"
        report_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n[5/5] Pipeline summary")
        print(f"  PDFs processed: {len(pdf_files)}")
        print(f"  Concepts created: {len(catalogue)}")
        print(f"  Structure failures: {len(structure_failures)}")
        print(f"  Provenance failures: {len(provenance_failures)}")
        print(f"  Text-quality findings: {len(text_findings)}")
        print(f"  Broken links: {len(link_errors)}")
        print(
            "  Missing reverse links (advisory): "
            f"{len(relationship_findings)}"
        )
        print(f"  Knowledge base: {final_index_path.parent}")
        print(f"  Report: {report_path}")

        if failures:
            print("\nPipeline completed with required validation failures.")
            return 1

        print("\nPipeline completed; required checks passed.")
        return 0

    except Exception:
        # Keep the staging directory for diagnosis, but never replace
        # the previous final knowledge base after an unexpected error.
        print(
            "\nPipeline failed. The previous unified knowledge base "
            "has not been replaced."
        )
        print(f"Inspect staging output if present: {staging_dir}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
