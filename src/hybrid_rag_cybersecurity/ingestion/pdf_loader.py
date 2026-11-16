"""PDF reader that extracts non-empty pages and page metadata."""

from pathlib import Path

import pymupdf

from hybrid_rag_cybersecurity.ingestion.document_schema import (
    DocumentRecord,
)


def load_pdf(file_path: str | Path) -> list[DocumentRecord]:
    """
    Extract text from a PDF document.

    Each PDF page becomes an individual DocumentRecord.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The supplied file is not a PDF.")

    documents = []

    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            document = DocumentRecord(
                text=text,
                metadata={
                    "source": path.name,
                    "source_path": str(path),
                    "file_type": "pdf",
                    "page_number": page_number,
                    "total_pages": len(pdf),
                },
            )

            documents.append(document)

    return documents
