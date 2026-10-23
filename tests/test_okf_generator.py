
from src.generation.okf_generator import write_okf_concepts


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

    import yaml

    ids = []
    for path in paths:
        markdown = path.read_text(encoding="utf-8")
        frontmatter = markdown.split("---", 2)[1]
        metadata = yaml.safe_load(frontmatter)
        ids.append(metadata["id"])

    assert ids == ["KC-JESS101-004", "KC-JESS101-005"]
