"""CSV reader that turns populated rows into document records."""

import csv
from pathlib import Path

from hybrid_rag_cybersecurity.ingestion.document_schema import (
    DocumentRecord,
)


def load_csv(file_path: str | Path) -> list[DocumentRecord]:
    """
    Extract CSV data.

    Each CSV row becomes an individual DocumentRecord.
    Column headers are retained in the extracted text.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("The supplied file is not a CSV document.")

    documents = []

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            return []

        for row_number, row in enumerate(reader, start=2):
            row_content = []

            for column, value in row.items():
                if value is None:
                    continue

                value = value.strip()

                if value:
                    row_content.append(f"{column}: {value}")

            if not row_content:
                continue

            text = "\n".join(row_content)

            documents.append(
                DocumentRecord(
                    text=text,
                    metadata={
                        "source": path.name,
                        "source_path": str(path),
                        "file_type": "csv",
                        "row_number": row_number,
                    },
                )
            )

    return documents
