
import json
from pathlib import Path


def save_extracted_pages(
    pdf_path: str | Path,
    pages: list[dict],
    output_dir: str | Path,
) -> Path:
    """Save extracted pages and source metadata as JSON."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{pdf_path.stem}.json"

    payload = {
        "source_file": pdf_path.name,
        "page_count": len(pages),
        "pages": pages,
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path
