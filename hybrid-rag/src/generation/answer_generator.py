import json
import re

from dataclasses import dataclass

import ollama

from src.fusion.ranker import FusedRecord


DEFAULT_LLM_MODEL = "qwen3:1.7b"


@dataclass(frozen=True)
class SourceCitation:
    evidence_id: str
    source_type: str
    description: str
    record_id: str


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    citations: tuple[SourceCitation, ...]


class AnswerGenerator:

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
    ):
        self.model = model

    @staticmethod
    def _format_source(
        evidence_id: str,
        record: FusedRecord,
    ) -> tuple[str, SourceCitation]:

        metadata = record.metadata
        source_type = metadata.get("source_type")

        if source_type == "pdf":
            document = metadata.get("document_name", "Unknown document")

            try:
                pages = json.loads(metadata.get("page_numbers", "[]"))
            except (TypeError, json.JSONDecodeError):
                pages = []

            description = (
                f"PDF: {document}, pages {pages}"
            )

        elif source_type == "okf":
            concept_id = metadata.get("concept_id", record.record_id)
            title = metadata.get("title", "Unknown concept")

            try:
                source_files = json.loads(
                    metadata.get("source_files", "[]")
                )
                source_pages = json.loads(
                    metadata.get("source_pages", "[]")
                )
            except (TypeError, json.JSONDecodeError):
                source_files = []
                source_pages = []

            description = (
                f"OKF concept: {title} ({concept_id}); "
                f"original sources: {source_files}; "
                f"pages: {source_pages}"
            )

        else:
            raise ValueError(
                f"Unsupported evidence source type: {source_type}"
            )

        citation = SourceCitation(
            evidence_id=evidence_id,
            source_type=source_type,
            description=description,
            record_id=record.record_id,
        )

        context = (
            f"[{evidence_id}]\n"
            f"Source: {description}\n"
            f"Content:\n{record.text}\n"
        )

        return context, citation

    def generate(
        self,
        question: str,
        evidence: list[FusedRecord],
    ) -> GeneratedAnswer:

        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not evidence:
            return GeneratedAnswer(
                answer=(
                    "Insufficient evidence: I could not find enough "
                    "information in the available documents to answer "
                    "this question."
                ),
                citations=(),
            )

        contexts = []
        citation_map = {}

        for index, record in enumerate(evidence, start=1):
            evidence_id = f"E{index}"

            context, citation = self._format_source(
                evidence_id,
                record,
            )

            contexts.append(context)
            citation_map[evidence_id] = citation

        system_prompt = """
You are a grounded question-answering assistant.

Answer the user's question using ONLY the supplied evidence.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts, figures, or explanations.
3. Every factual claim must have an evidence reference.
4. Cite evidence using its exact identifier, such as [E1].
5. You may cite multiple references where appropriate.
6. If the evidence is insufficient, explicitly say so.
7. Do not invent or modify evidence identifiers.
8. Keep the answer clear and concise.
9. Do not cite a source unless its content supports the claim.
"""

        user_prompt = f"""
QUESTION:
{question}

RETRIEVED EVIDENCE:

{"".join(contexts)}

Provide a grounded answer with evidence references.
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": 0,
            },
        )

        answer = response.message.content.strip()

        if not answer:
            raise RuntimeError("Ollama returned an empty answer")

        cited_ids = list(dict.fromkeys(
            re.findall(r"\[(E\d+)\]", answer)
        ))

        invalid_ids = [
            evidence_id
            for evidence_id in cited_ids
            if evidence_id not in citation_map
        ]

        if invalid_ids:
            raise ValueError(
                f"LLM returned invalid evidence references: {invalid_ids}"
            )

        citations = tuple(
            citation_map[evidence_id]
            for evidence_id in cited_ids
        )

        return GeneratedAnswer(
            answer=answer,
            citations=citations,
        )