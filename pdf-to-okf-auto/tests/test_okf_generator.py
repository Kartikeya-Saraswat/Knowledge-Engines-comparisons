import json
from types import SimpleNamespace

import yaml

from src.generation.okf_generator import (
    generate_concepts,
    write_okf_concepts,
)


def mock_ollama_response(monkeypatch, concepts):
    response_content = json.dumps({"concepts": concepts})

    def fake_chat(**kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(content=response_content)
        )

    monkeypatch.setattr(
        "src.generation.okf_generator.ollama.chat",
        fake_chat,
    )


def test_concept_ids_follow_start_index(tmp_path):
    concepts = [
        {
            "title": "First concept",
            "description": "First description",
            "tags": [],
            "content": "First content.",
            "source_pages": [1],
        },
        {
            "title": "Second concept",
            "description": "Second description",
            "tags": [],
            "content": "Second content.",
            "source_pages": [2],
        },
    ]

    paths = write_okf_concepts(
        concepts,
        source_file="jess101.pdf",
        output_dir=tmp_path,
        start_index=4,
    )

    ids = []
    for path in paths:
        markdown = path.read_text(encoding="utf-8")
        frontmatter = markdown.split("---", 2)[1]
        metadata = yaml.safe_load(frontmatter)
        ids.append(metadata["id"])

    assert ids == ["KC-JESS101-004", "KC-JESS101-005"]


def test_generator_accepts_verified_evidence(monkeypatch):
    quote = (
        "Resource planning is a complex process which involves "
        "identification and inventory of resources across the regions."
    )

    mock_ollama_response(monkeypatch, [{
        "title": "Resource Planning in India",
        "description": "The process of planning resource development.",
        "tags": ["resources"],
        "content": "Resource planning involves identifying resources.",
        "evidence": [
            {"page_number": 3, "quote": quote}
        ],
    }])

    pages = [
        {"page_number": 2, "text": "Resource planning is essential."},
        {"page_number": 3, "text": quote},
    ]

    concepts = generate_concepts(pages, "jess101.pdf")

    assert len(concepts) == 1
    assert concepts[0]["source_pages"] == [3]
    assert concepts[0]["evidence"][0]["page_number"] == 3


def test_generator_rejects_unverified_quote(monkeypatch):
    mock_ollama_response(monkeypatch, [{
        "title": "Resource Planning in India",
        "description": "A description.",
        "tags": [],
        "content": "Some generated content.",
        "evidence": [
            {
                "page_number": 3,
                "quote": (
                    "This passage does not appear in the supplied "
                    "textbook page and should be rejected."
                ),
            }
        ],
    }])

    pages = [
        {
            "page_number": 3,
            "text": "Resource planning involves identifying resources.",
        }
    ]

    concepts = generate_concepts(pages, "jess101.pdf")

    assert concepts == []


def test_generator_rejects_evidence_from_wrong_page(monkeypatch):
    quote = (
        "Resource planning is a complex process which involves "
        "identification and inventory of resources across the regions."
    )

    mock_ollama_response(monkeypatch, [{
        "title": "Resource Planning in India",
        "description": "A description.",
        "tags": [],
        "content": "Some generated content.",
        "evidence": [
            {"page_number": 1, "quote": quote}
        ],
    }])

    pages = [
        {
            "page_number": 1,
            "text": "Everything available in our environment is a resource.",
        },
        {
            "page_number": 3,
            "text": quote,
        },
    ]

    concepts = generate_concepts(pages, "jess101.pdf")

    assert concepts == []


def test_generator_uses_multiple_verified_pages(monkeypatch):
    quote_page_2 = (
        "Therefore, resource planning is essential for sustainable "
        "existence of all forms of life."
    )
    quote_page_3 = (
        "Resource planning is a complex process which involves "
        "identification and inventory of resources across the regions."
    )

    mock_ollama_response(monkeypatch, [{
        "title": "Resource Planning",
        "description": "Resource planning supports sustainable development.",
        "tags": [],
        "content": "Resource planning supports sustainable development.",
        "evidence": [
            {"page_number": 2, "quote": quote_page_2},
            {"page_number": 3, "quote": quote_page_3},
        ],
    }])

    pages = [
        {"page_number": 2, "text": quote_page_2},
        {"page_number": 3, "text": quote_page_3},
    ]

    concepts = generate_concepts(pages, "jess101.pdf")

    assert len(concepts) == 1
    assert concepts[0]["source_pages"] == [2, 3]


def test_writer_stores_evidence_in_metadata(tmp_path):
    quote = (
        "Resource planning is a complex process which involves "
        "identification and inventory of resources across the regions."
    )

    concepts = [{
        "title": "Resource Planning",
        "description": "A description.",
        "tags": [],
        "content": "Resource planning content.",
        "source_pages": [3],
        "evidence": [
            {"page_number": 3, "quote": quote}
        ],
    }]

    paths = write_okf_concepts(
        concepts,
        "jess101.pdf",
        tmp_path,
    )

    metadata = yaml.safe_load(
        paths[0].read_text(encoding="utf-8").split("---", 2)[1]
    )

    assert metadata["evidence"][0]["page_number"] == 3
    assert metadata["sources"] == ["jess101.pdf, pages 3"]