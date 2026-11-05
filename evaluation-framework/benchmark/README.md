# Benchmark Design

## Purpose
The benchmark provides a common ground truth for evaluating PDF RAG, OKF-only retrieval, and Hybrid RAG.

## Question structure

Each question contains:

- id: Unique question identifier.
- question: The question asked of the systems.
- type: factual, conceptual, multi_hop, or unanswerable.
- answerable: Whether the supplied corpus contains sufficient evidence.
- expected_answer: A manually verified answer outline.
- evidence_units: The distinct facts required to answer the question.

## Evidence unit structure

Each evidence unit contains:

- id: Unique evidence identifier within the question.
- relevance: Importance of the evidence (1–3).
- pdf_anchors: PDF document, page numbers, and exact supporting quotation.
- okf_concept_ids: OKF concepts representing the same evidence.

## Relevance grading

- 3: Essential evidence required to answer the question.
- 2: Important supporting evidence.
- 1: Useful contextual evidence.

## Example

{
  "id": "Q001",
  "question": "Example question",
  "type": "factual",
  "answerable": true,
  "expected_answer": "Manually verified answer outline",
  "evidence_units": [
    {
      "id": "EU1",
      "relevance": 3,
      "pdf_anchors": [
        {
          "document": "jess101.pdf",
          "pages": [1],
          "quote": "Exact quotation copied from the textbook."
        }
      ],
      "okf_concept_ids": ["KC-JESS101-001"]
    }
  ]
}

## Ground-truth rules

1. Questions and answers must be manually verified against the source textbooks.
2. PDF quotations must be copied exactly from the extracted textbook text.
3. Page numbers must refer to the original PDF pages.
4. OKF concept IDs must exist in the generated OKF corpus.
5. One evidence unit represents one distinct fact, not one retrieved chunk.
6. Multiple PDF anchors or OKF concepts may represent the same evidence unit.
7. Unanswerable questions must have answerable=false and an empty evidence_units list.
8. Do not use generated answers as ground truth without manual verification.

## Initial benchmark target

Start with 27 questions: approximately three per textbook.

Include factual, conceptual, multi-hop, and unanswerable questions.
