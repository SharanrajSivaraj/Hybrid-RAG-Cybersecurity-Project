"""Text-file reader that creates a document record for non-empty input."""

from pathlib import Path

from hybrid_rag_cybersecurity.ingestion.document_schema import (
    DocumentRecord,
)


def load_txt(file_path: str | Path) -> list[DocumentRecord]:
    """
    Extract text from a plain-text file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"TXT file not found: {path}")

    if path.suffix.lower() != ".txt":
        raise ValueError("The supplied file is not a TXT document.")

    text = path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    ).strip()

    if not text:
        return []

    return [
        DocumentRecord(
            text=text,
            metadata={
                "source": path.name,
                "source_path": str(path),
                "file_type": "txt",
            },
        )
    ]
