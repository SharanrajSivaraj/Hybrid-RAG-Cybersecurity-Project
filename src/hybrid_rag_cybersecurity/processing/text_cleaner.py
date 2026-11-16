"""Normalization of extracted text while retaining useful line breaks."""

import re

from hybrid_rag_cybersecurity.ingestion.document_schema import (
    DocumentRecord,
)


def clean_text(document: DocumentRecord) -> DocumentRecord:
    """
    Clean extracted document text while preserving
    meaningful paragraph and line boundaries.
    """

    text = document.text

    # Normalise Windows and Mac line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove trailing spaces from individual lines.
    lines = [line.strip() for line in text.split("\n")]

    text = "\n".join(lines)

    # Replace 3 or more consecutive newlines with 2.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove excessive spaces and tabs.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return DocumentRecord(
        text=text.strip(),
        metadata=document.metadata.copy(),
    )
