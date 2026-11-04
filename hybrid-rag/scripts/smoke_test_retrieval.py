"""Check retrieval against questions drawn from different textbooks."""

from pathlib import Path
import logging

from src.pipeline import HybridRAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = PROJECT_ROOT / "data/input/sample"
OKF_DIR = PROJECT_ROOT / "data/input/unified_okf"

QUESTIONS = (
    ("jess101.pdf", "What is resource planning in India?"),
    ("jess102.pdf", "How are forests and wildlife conserved in India?"),
    ("jess103.pdf", "What are the benefits and impacts of multipurpose river projects?"),
    ("jess104.pdf", "What conditions are needed for rice cultivation?"),
    ("jess105.pdf", "Where are major iron ore producing regions in India?"),
    ("jess106.pdf", "What factors influence the location of industries?"),
    ("jess107.pdf", "How has the railway network developed in India?"),
    ("jess1a1.pdf", "What environmental organizations work in India?"),
    ("jess1ps.pdf", "What is the National Curriculum Framework 2005?"),
)


def main() -> None:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    pipeline = HybridRAGPipeline()
    print("Clearing and rebuilding the index from all source files...")
    counts = pipeline.rebuild_index(PDF_DIR, OKF_DIR)
    print("Indexed:", counts)

    hits = 0
    for expected_document, question in QUESTIONS:
        print(f"\nQuestion ({expected_document}): {question}")
        results = pipeline.retrieve_hybrid(
            question,
            pdf_top_k=5,
            okf_top_k=5,
            final_top_k=8,
        )

        expected_ranks = []
        for rank, item in enumerate(results, start=1):
            source = item.metadata.get("source_type")
            if source == "pdf":
                document = item.metadata.get("document_name", "")
            else:
                source_files = item.metadata.get("source_files", "")
                document = str(source_files)

            matches = expected_document.casefold() in document.casefold()
            if matches:
                expected_ranks.append(rank)
            print(
                f"  {rank}. {source} id={item.record_id} "
                f"rrf={item.rrf_score:.6f} expected_source={matches}"
            )
            print(f"     {item.text[:260].replace(chr(10), ' ')}")

        found = bool(expected_ranks)
        hits += int(found)
        print(f"Expected textbook ranks in top 8: {expected_ranks or 'none'}")

    print(f"\nExpected textbook found for {hits}/{len(QUESTIONS)} questions.")
    print("Review the ranked excerpts above to assess relevance and provenance.")


if __name__ == "__main__":
    main()
