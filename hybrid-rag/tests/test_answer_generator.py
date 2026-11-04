import json
from types import SimpleNamespace

import pytest

from src.fusion.ranker import FusedRecord
from src.generation.answer_generator import AnswerGenerator


def make_pdf_evidence():
    return FusedRecord(
        record_id="pdf-001",
        text="Resource planning involves identifying and developing resources.",
        metadata={
            "source_type": "pdf",
            "document_name": "jess101.pdf",
            "page_numbers": json.dumps([3]),
        },
        rrf_score=0.03,
        source_ranks={"pdf": 1},
    )


def make_okf_evidence():
    return FusedRecord(
        record_id="KC-JESS101-001",
        text="Resource Planning in India involves identification and inventory of resources.",
        metadata={
            "source_type": "okf",
            "concept_id": "KC-JESS101-001",
            "title": "Resource Planning in India",
            "source_files": json.dumps(["jess101.pdf"]),
            "source_pages": json.dumps([3]),
        },
        rrf_score=0.03,
        source_ranks={"okf": 1},
    )


def mock_ollama(monkeypatch, answer):
    import src.generation.answer_generator as module

    monkeypatch.setattr(
        module.ollama,
        "chat",
        lambda **kwargs: SimpleNamespace(
            message=SimpleNamespace(content=answer)
        ),
    )


def test_generate_answer_with_pdf_citation(monkeypatch):
    mock_ollama(
        monkeypatch,
        "Resource planning involves identifying resources. [E1]",
    )

    generator = AnswerGenerator()

    result = generator.generate(
        "What is resource planning?",
        [make_pdf_evidence()],
    )

    assert "identifying resources" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].source_type == "pdf"
    assert "jess101.pdf" in result.citations[0].description


def test_generate_answer_with_okf_citation(monkeypatch):
    mock_ollama(
        monkeypatch,
        "Resource planning involves identifying resources. [E1]",
    )

    generator = AnswerGenerator()

    result = generator.generate(
        "What is resource planning?",
        [make_okf_evidence()],
    )

    assert len(result.citations) == 1
    assert result.citations[0].source_type == "okf"
    assert result.citations[0].record_id == "KC-JESS101-001"


def test_generate_answer_with_multiple_sources(monkeypatch):
    mock_ollama(
        monkeypatch,
        "Resource planning involves identifying resources. [E1] [E2]",
    )

    generator = AnswerGenerator()

    result = generator.generate(
        "What is resource planning?",
        [
            make_pdf_evidence(),
            make_okf_evidence(),
        ],
    )

    assert len(result.citations) == 2
    assert result.citations[0].source_type == "pdf"
    assert result.citations[1].source_type == "okf"


def test_no_evidence_returns_insufficient_evidence():
    generator = AnswerGenerator()

    result = generator.generate(
        "What is resource planning?",
        [],
    )

    assert "insufficient evidence" in result.answer.lower()
    assert result.citations == ()


def test_empty_question_is_rejected():
    generator = AnswerGenerator()

    with pytest.raises(ValueError):
        generator.generate("   ", [make_pdf_evidence()])


def test_invalid_citation_is_rejected(monkeypatch):
    mock_ollama(
        monkeypatch,
        "Resource planning involves identifying resources. [E99]",
    )

    generator = AnswerGenerator()

    with pytest.raises(ValueError, match="invalid evidence references"):
        generator.generate(
            "What is resource planning?",
            [make_pdf_evidence()],
        )


def test_empty_llm_response_is_rejected(monkeypatch):
    mock_ollama(monkeypatch, "")

    generator = AnswerGenerator()

    with pytest.raises(RuntimeError, match="empty answer"):
        generator.generate(
            "What is resource planning?",
            [make_pdf_evidence()],
        )