"""File-extension dispatch for supported document readers."""

from pathlib import Path

from hybrid_rag_cybersecurity.ingestion.csv_loader import load_csv
from hybrid_rag_cybersecurity.ingestion.document_schema import (
    DocumentRecord,
)
from hybrid_rag_cybersecurity.ingestion.docx_loader import load_docx
from hybrid_rag_cybersecurity.ingestion.pdf_loader import load_pdf
from hybrid_rag_cybersecurity.ingestion.txt_loader import load_txt
from hybrid_rag_cybersecurity.ingestion.xlsx_loader import load_xlsx

SUPPORTED_LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
    ".csv": load_csv,
    ".xlsx": load_xlsx,
}


def load_file(file_path: str | Path) -> list[DocumentRecord]:
    """
    Automatically detect the file format
    and call the appropriate document loader.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()

    loader = SUPPORTED_LOADERS.get(extension)

    if loader is None:
        raise ValueError(f"Unsupported file format: {extension}")

    return loader(path)
