"""Manual smoke test for loading supported source-document formats."""

from collections import Counter
from pathlib import Path

from hybrid_rag_cybersecurity.ingestion import load_directory


# Run the workflow described in this module overview.
def main() -> None:

    project_root = Path(__file__).resolve().parents[1]

    raw_data_directory = project_root / "data" / "raw"

    print("=" * 60)
    print("SENTINEL CYBERSECURITY - DOCUMENT INGESTION")
    print("=" * 60)

    documents, failures = load_directory(raw_data_directory)

    print("\nRECORD BREAKDOWN")
    print("-" * 60)

    file_type_counts = Counter(document.metadata["file_type"] for document in documents)

    for file_type, count in sorted(file_type_counts.items()):
        print(f"{file_type.upper()}: {count} records")

    source_counts = Counter(document.metadata["source"] for document in documents)

    print("\nRECORDS BY SOURCE FILE")
    print("-" * 60)

    for source, count in sorted(source_counts.items()):
        print(f"{source}: {count} records")

    print(f"\nTotal extracted records: {len(documents)}")

    print("\nEXTRACTED DOCUMENTS")
    print("-" * 60)

    for index, document in enumerate(documents[:10], start=1):
        print(f"\nRecord {index}")

        print(f"Source: {document.metadata['source']}")

        print(f"File type: {document.metadata['file_type']}")

        print(f"Metadata: {document.metadata}")

        print(f"Text preview: {document.text[:200]}")

    if len(documents) > 10:
        print(f"\n... and {len(documents) - 10} more records.")

    print("\nINGESTION FAILURES")
    print("-" * 60)

    if not failures:
        print("No ingestion failures.")
    else:
        for failure in failures:
            print(f"File: {failure['file']}")
            print(f"Error: {failure['error']}")
            print("-" * 40)

    print("\nINGESTION COMPLETED")


if __name__ == "__main__":
    main()
