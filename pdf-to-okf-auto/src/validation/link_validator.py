
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


MARKDOWN_LINK_PATTERN = re.compile(
    r"!?\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))(?:\s+[^)]*)?\)"
)


def validate_markdown_links(
    markdown_dir: str | Path,
) -> list[str]:
    """Return errors for broken local Markdown file links."""
    markdown_dir = Path(markdown_dir)

    if not markdown_dir.exists():
        raise FileNotFoundError(
            f"Markdown directory not found: {markdown_dir}"
        )

    errors = []

    for source_path in sorted(markdown_dir.glob("*.md")):
        content = source_path.read_text(encoding="utf-8")

        for match in MARKDOWN_LINK_PATTERN.finditer(content):
            raw_target = match.group(1) or match.group(2)
            target = unquote(raw_target.strip())

            if not target or target.startswith("#"):
                continue

            parsed = urlsplit(target)

            # External URLs and non-file schemes aren't local links.
            if parsed.scheme or parsed.netloc:
                continue

            target_path = parsed.path
            if not target_path:
                continue

            resolved_target = (
                source_path.parent / target_path
            ).resolve()

            if not resolved_target.is_file():
                errors.append(
                    f"{source_path.name}: broken link '{target}'"
                )

    return errors
