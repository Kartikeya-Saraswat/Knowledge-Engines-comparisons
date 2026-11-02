from types import SimpleNamespace

import pytest

from src.generation.answer_generator import (
    DEFAULT_OLLAMA_MODEL,
    INSUFFICIENT_CONTEXT_ANSWER,
    AnswerGenerator,
)
from src.retrieval.vector_store import RetrievedChunk


class FakeOllamaClient:
    def __init__(self, content="The answer is 42 [report.pdf, p. 3]."):
        self.content = content
        self.call = None

    def chat(self, *, model, messages):
        self.call = {"model": model, "messages": messages}
        return SimpleNamespace(
            message=SimpleNamespace(content=self.content)
        )


def make_chunk(
    chunk_id="chunk-1",
    text="The report states the answer is 42.",
    document_name="report.pdf",
    pages=(3,),
):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        document_name=document_name,
        page_numbers=pages,
        distance=0.1,
    )


def test_uses_qwen_model_and_returns_answer_with_citations():
    client = FakeOllamaClient()
    generator = AnswerGenerator(client=client)

    answer = generator.generate("What is the answer?", [make_chunk()])

    assert answer == "The answer is 42 [report.pdf, p. 3]."
    assert client.call["model"] == "qwen3:1.7b"
    assert generator.model_name == DEFAULT_OLLAMA_MODEL
    assert client.call["messages"][0]["role"] == "system"


def test_prompt_contains_retrieved_text_document_and_physical_pages():
    client = FakeOllamaClient()
    generator = AnswerGenerator(client=client)

    generator.generate(
        "Summarize the finding.",
        [make_chunk(text="A finding across two pages.", pages=(2, 5))],
    )

    prompt = client.call["messages"][1]["content"]
    assert "report.pdf" in prompt
    assert "Physical page(s): 2, 5" in prompt
    assert "A finding across two pages." in prompt
    assert "Do not invent sources or page numbers." in client.call["messages"][0]["content"]


def test_no_chunks_returns_fallback_without_calling_model():
    client = FakeOllamaClient()
    generator = AnswerGenerator(client=client)

    answer = generator.generate("What happened?", [])

    assert answer == INSUFFICIENT_CONTEXT_ANSWER
    assert client.call is None


def test_blank_chunk_text_is_treated_as_missing_context():
    client = FakeOllamaClient()
    generator = AnswerGenerator(client=client)

    answer = generator.generate("What happened?", [make_chunk(text="  ")])

    assert answer == INSUFFICIENT_CONTEXT_ANSWER
    assert client.call is None


def test_blank_question_is_rejected():
    with pytest.raises(ValueError, match="question cannot be empty"):
        AnswerGenerator(client=FakeOllamaClient()).generate("  ", [make_chunk()])


def test_empty_model_response_uses_insufficient_context_fallback():
    generator = AnswerGenerator(client=FakeOllamaClient(content="  "))

    assert generator.generate("What happened?", [make_chunk()]) == INSUFFICIENT_CONTEXT_ANSWER
