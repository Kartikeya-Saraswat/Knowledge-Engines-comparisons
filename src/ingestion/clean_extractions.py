
import json
from pathlib import Path

from src.ingestion.text_cleaner import clean_extracted_text


def clean_extraction_file(
    input_path: str | Path,
    output_dir: str | Path,
) -> Path:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))

    for page in payload["pages"]:
        page["text"] = clean_extracted_text(page.get("text", ""))

    output_path = output_dir / input_path.name
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path
