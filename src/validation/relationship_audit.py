
from pathlib import Path
import re


LINK_PATTERN = re.compile(r"^- \[([^\]]+)\]\(([^)]+)\)\s*$", re.MULTILINE)


def audit_relationships(catalogue: list[dict]) -> list[dict]:
    """Report existing concept links that have no reverse link."""
    filename_to_id = {
        Path(concept["filename"]).name: concept["id"]
        for concept in catalogue
    }

    links_by_id: dict[str, set[str]] = {
        concept["id"]: set()
        for concept in catalogue
    }

    for concept in catalogue:
        source_id = concept["id"]
        path = Path(concept["path"])
        markdown = path.read_text(encoding="utf-8")

        section = markdown.split("## Related concepts", 1)
        if len(section) < 2:
            continue

        related_section = section[1].split("\n## ", 1)[0]

        for _, target_filename in LINK_PATTERN.findall(related_section):
            target_id = filename_to_id.get(Path(target_filename).name)
            if target_id and target_id != source_id:
                links_by_id[source_id].add(target_id)

    findings = []

    for source_id, target_ids in links_by_id.items():
        for target_id in sorted(target_ids):
            if source_id not in links_by_id.get(target_id, set()):
                findings.append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "issue": "missing_reverse_link",
                })

    return findings
