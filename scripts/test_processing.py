"""Manual smoke test for text cleaning and document chunking."""

from collections import Counter
from pathlib import Path

from hybrid_rag_cybersecurity.ingestion import load_directory
from hybrid_rag_cybersecurity.processing.chunker import chunk_documents
from hybrid_rag_cybersecurity.processing.text_cleaner import clean_text


# Run the workflow described in this module overview.
def main() -> None:

    project_root = Path(__file__).resolve().parents[1]

    raw_data_directory = project_root / "data" / "raw"

    documents, failures = load_directory(raw_data_directory)

    cleaned_documents = [clean_text(document) for document in documents]

    chunked_documents = chunk_documents(
        cleaned_documents,
        chunk_size=1200,
        chunk_overlap=200,
    )

    print("=" * 60)
    print("SENTINEL CYBERSECURITY - DOCUMENT PROCESSING")
    print("=" * 60)

    print(f"\nExtracted records: {len(documents)}")
    print(f"Cleaned records: {len(cleaned_documents)}")
    print(f"Generated chunks: {len(chunked_documents)}")

    print("\nCHUNKS BY FILE TYPE")
    print("-" * 60)

    counts = Counter(document.metadata["file_type"] for document in chunked_documents)

    for file_type, count in sorted(counts.items()):
        print(f"{file_type.upper()}: {count} chunks")

    print("\nSAMPLE CHUNKS")
    print("-" * 60)

    for index, document in enumerate(chunked_documents[:5], start=1):
        print(f"\nChunk {index}")
        print(f"Source: {document.metadata['source']}")
        print(f"Chunk index: {document.metadata['chunk_index']}")
        print(f"Chunk size: {document.metadata['chunk_size']}")
        print(f"Text: {document.text[:300]}")

    print("\nINGESTION FAILURES:", len(failures))
    print("\nPROCESSING COMPLETED")


if __name__ == "__main__":
    main()
