"""Content fingerprints for detecting changes to conversion and retrieval inputs."""

from hashlib import sha256
from pathlib import Path


def fingerprint_files(paths, root: Path) -> str:
    digest = sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            file_digest = sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(block)
        digest.update(file_digest.digest())
    return digest.hexdigest()


def fingerprint_concepts(directory: str | Path) -> str:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"OKF directory does not exist: {directory}")
    return fingerprint_files(directory.glob("*.md"), directory)
