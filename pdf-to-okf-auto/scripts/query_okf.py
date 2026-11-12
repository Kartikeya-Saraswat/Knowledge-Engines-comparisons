"""Index saved OKF concepts and answer one question or start interactive QA."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import OKFRAGPipeline


def answer_questions(pipeline, *, question=None, interactive=False, top_k=5):
    def ask(text):
        result = pipeline.answer_question(text, top_k=top_k)
        print("\nAnswer:")
        print(result.answer)
        for citation in result.citations:
            print(
                f"[{citation.evidence_id}] {citation.title} ({citation.record_id}); "
                f"PDF sources: {', '.join(citation.source_files)}; "
                f"physical pages: {', '.join(map(str, citation.source_pages))}"
            )

    if question:
        ask(question)
    if interactive:
        print("\nOKF QA is ready. Type 'exit' to quit.")
        while True:
            try:
                text = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.casefold() in {"exit", "quit"}:
                break
            if text:
                ask(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--okf-dir", default="data/output/ollama_full_run/unified")
    parser.add_argument("--index-dir", default="data/chroma_db")
    parser.add_argument("--question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--no-index", action="store_true",
                        help="Query an existing index without re-embedding concepts.")
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    if args.question is not None and not args.question.strip():
        parser.error("--question cannot be empty")

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    index_dir = resolve(args.index_dir)
    if args.no_index and not (index_dir / "chroma.sqlite3").is_file():
        parser.error("Index not found. Run without --no-index first.")
    pipeline = OKFRAGPipeline(index_dir, answer_model=args.answer_model)
    if not args.no_index:
        pipeline.ensure_okf_directory(resolve(args.okf_dir))
    elif pipeline.vector_store.collection.count() == 0:
        parser.error("OKF collection is empty. Run without --no-index first.")
    answer_questions(pipeline, question=args.question,
                     interactive=args.question is None, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
