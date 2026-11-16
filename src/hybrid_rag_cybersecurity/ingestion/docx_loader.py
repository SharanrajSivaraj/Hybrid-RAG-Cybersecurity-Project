"""DOCX reader for paragraphs and table rows."""

from pathlib import Path

from docx import Document

from hybrid_rag_cybersecurity.ingestion.document_schema import (
    DocumentRecord,
)


def load_docx(file_path: str | Path) -> list[DocumentRecord]:
    """
    Extract paragraphs and tables from a DOCX document.

    The entire document becomes one DocumentRecord initially.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"DOCX file not found: {path}")

    if path.suffix.lower() != ".docx":
        raise ValueError("The supplied file is not a DOCX document.")

    doc = Document(path)

    content = []

    # Extract paragraphs
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        if text:
            content.append(text)

    # Extract tables
    for table in doc.tables:
        for row in table.rows:
            row_values = [cell.text.strip() for cell in row.cells]

            row_text = " | ".join(row_values)

            if row_text:
                content.append(row_text)

    full_text = "\n\n".join(content)

    if not full_text.strip():
        return []

    return [
        DocumentRecord(
            text=full_text,
            metadata={
                "source": path.name,
                "source_path": str(path),
                "file_type": "docx",
            },
        )
    ]
