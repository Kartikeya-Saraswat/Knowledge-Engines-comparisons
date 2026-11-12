"""Verify benchmark quotes with PDF-extraction-tolerant normalization.

Run from evaluation-framework:
    ..\pdf-to-okf-auto\.venv\Scripts\python.exe .\scripts\verify_pdf_evidence.py

This checks whether the quote's words occur on its cited page(s), while ignoring
formatting-only differences such as punctuation, hyphenation, curly quotes,
and repeated OCR/PDF extraction tokens (3+ consecutive repetitions).
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = ROOT / "benchmark" / "questions.json"
PDF_DIR = ROOT.parent / "pdf-to-okf-auto" / "data" / "input" / "sample"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("\u00ad", "")
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"[\u2018\u2019\u02bc`´]", "'", text)
    # Join line-end hyphenation and compound names split by extraction.
    text = re.sub(r"(?<=\w)-\s*(?=\w)", "", text)
    words = re.findall(r"\w+", text, flags=re.UNICODE)

    # Some PDF extraction paths duplicate words from figures/text layers.
    # Collapse only runs of 3+ identical adjacent tokens, not normal repetition.
    collapsed = []
    i = 0
    while i < len(words):
        j = i + 1
        while j < len(words) and words[j] == words[i]:
            j += 1
        collapsed.append(words[i])
        i = j
    return "".join(collapsed)


def main() -> int:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    cache = {}
    total = 0
    matched = 0
    issues = []

    for q in questions:
        for unit in q.get("evidence_units", []):
            for index, anchor in enumerate(unit.get("pdf_anchors", []), 1):
                total += 1
                document = anchor.get("document")
                pages = anchor.get("pages", [])
                quote = anchor.get("quote", "")
                path = PDF_DIR / str(document)

                if not path.is_file():
                    issues.append((q["id"], unit.get("id"), index,
                                   f"PDF not found: {path}"))
                    continue
                if document not in cache:
                    with pymupdf.open(path) as pdf:
                        cache[document] = [
                            normalize(page.get_text("text")) for page in pdf
                        ]

                page_texts = cache[document]
                if not pages or any(
                    not isinstance(p, int) or p < 1 or p > len(page_texts)
                    for p in pages
                ):
                    issues.append((q["id"], unit.get("id"), index,
                                   f"Invalid page list: {pages}"))
                    continue

                normalized_quote = normalize(quote)
                cited_text = "".join(page_texts[p - 1] for p in pages)

                if normalized_quote and normalized_quote in cited_text:
                    matched += 1
                else:
                    elsewhere = [
                        p for p, page_text in enumerate(page_texts, 1)
                        if normalized_quote and normalized_quote in page_text
                    ]
                    issues.append((
                        q["id"], unit.get("id"), index,
                        f"Not found on cited page(s) {pages}; "
                        f"exact normalized match elsewhere: {elsewhere or 'not found'}"
                    ))

    print(f"Questions checked: {len(questions)}")
    print(f"PDF evidence anchors checked: {total}")
    print(f"Matches after formatting normalization: {matched}")
    print(f"Remaining anchors for manual review: {len(issues)}")
    for qid, unit_id, index, detail in issues:
        print(f"- {qid} / {unit_id} / anchor {index}: {detail}")

    if issues:
        print("\nReview remaining mismatches against the original PDF page.")
        return 1
    print("PDF evidence verification PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
