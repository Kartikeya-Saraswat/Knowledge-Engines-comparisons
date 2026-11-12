"""Print context for PDF evidence anchors that failed exact quote matching.

Run from evaluation-framework:
    ..\pdf-to-okf-auto\.venv\Scripts\python.exe .\scripts\diagnose_pdf_evidence.py
"""
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = ROOT / "benchmark" / "questions.json"
PDF_DIR = ROOT.parent / "pdf-to-okf-auto" / "data" / "input" / "sample"


def norm(s):
    s = unicodedata.normalize("NFKC", s).casefold().replace("\u00ad", "")
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)
    s = re.sub(r"(?<=\w)-\s+(?=\w)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    cache = {}
    mismatch_count = 0

    for q in questions:
        for unit in q.get("evidence_units", []):
            for i, anchor in enumerate(unit.get("pdf_anchors", []), 1):
                doc_name = anchor["document"]
                pages = anchor["pages"]
                quote = anchor["quote"]
                path = PDF_DIR / doc_name

                if doc_name not in cache:
                    with pymupdf.open(path) as doc:
                        cache[doc_name] = [p.get_text("text") for p in doc]

                page_texts = cache[doc_name]
                cited_text = "\n".join(page_texts[p - 1] for p in pages)
                nq, nt = norm(quote), norm(cited_text)
                if nq and nq in nt:
                    continue

                mismatch_count += 1
                matcher = SequenceMatcher(None, nq, nt, autojunk=False)
                match = matcher.find_longest_match(0, len(nq), 0, len(nt))
                start = max(0, match.b - 220)
                end = min(len(nt), match.b + match.size + 220)
                excerpt = nt[start:end] if match.size else nt[:700]

                print("=" * 88)
                print(f"{q['id']} | {unit.get('id')} | anchor {i}")
                print(f"PDF: {doc_name} | cited physical page(s): {pages}")
                print(f"Quote ({len(quote)} chars):\n{quote}")
                print(
                    f"\nLongest matching sequence: {match.size} chars "
                    f"({match.size / max(1, len(nq)):.0%} of normalized quote)"
                )
                print(f"Nearby extracted page text:\n{excerpt}")

    print("=" * 88)
    print(f"Total anchors needing manual review: {mismatch_count}")


if __name__ == "__main__":
    main()
