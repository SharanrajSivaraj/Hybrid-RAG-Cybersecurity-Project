"""Excel reader that turns populated worksheet rows into records."""

from pathlib import Path

from openpyxl import load_workbook

from hybrid_rag_cybersecurity.ingestion.document_schema import (
    DocumentRecord,
)


def load_xlsx(file_path: str | Path) -> list[DocumentRecord]:
    """
    Extract Excel worksheet data.

    Each non-empty row becomes an individual DocumentRecord.
    Worksheet names and row numbers are preserved.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"XLSX file not found: {path}")

    if path.suffix.lower() != ".xlsx":
        raise ValueError("The supplied file is not an XLSX document.")

    documents = []

    workbook = load_workbook(
        filename=path,
        read_only=True,
        data_only=True,
    )

    try:
        for worksheet in workbook.worksheets:
            rows = worksheet.iter_rows(values_only=True)

            headers = next(rows, None)

            if headers is None:
                continue

            headers = [
                str(header).strip() if header is not None else f"column_{index + 1}"
                for index, header in enumerate(headers)
            ]

            for row_number, row in enumerate(rows, start=2):
                row_content = []

                for header, value in zip(headers, row, strict=False):
                    if value is None:
                        continue

                    value = str(value).strip()

                    if value:
                        row_content.append(f"{header}: {value}")

                if not row_content:
                    continue

                text = "\n".join(row_content)

                documents.append(
                    DocumentRecord(
                        text=text,
                        metadata={
                            "source": path.name,
                            "source_path": str(path),
                            "file_type": "xlsx",
                            "sheet_name": worksheet.title,
                            "row_number": row_number,
                        },
                    )
                )

    finally:
        workbook.close()

    return documents
