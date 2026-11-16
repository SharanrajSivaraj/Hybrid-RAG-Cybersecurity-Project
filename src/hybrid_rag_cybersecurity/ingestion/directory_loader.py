"""Recursive loader that collects supported files and records failures."""

from pathlib import Path

from hybrid_rag_cybersecurity.ingestion.document_schema import (
    DocumentRecord,
)
from hybrid_rag_cybersecurity.ingestion.loader import (
    SUPPORTED_LOADERS,
    load_file,
)


def load_directory(
    directory_path: str | Path,
) -> tuple[list[DocumentRecord], list[dict[str, str]]]:
    """
    Recursively load all supported documents from a directory.

    Returns:
        A tuple containing:
        1. Successfully extracted documents.
        2. A list of files that failed to load.
    """

    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory: {directory}")

    documents = []
    failures = []

    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_LOADERS:
            continue

        try:
            extracted = load_file(file_path)

            documents.extend(extracted)

        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "file": str(file_path),
                    "error": str(error),
                }
            )

    return documents, failures
