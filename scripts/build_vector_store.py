"""Command-line workflow that loads, chunks, embeds, and stores project documents."""

from pathlib import Path

from hybrid_rag_cybersecurity.ingestion import load_directory
from hybrid_rag_cybersecurity.processing.chunker import chunk_documents
from hybrid_rag_cybersecurity.processing.text_cleaner import clean_text
from hybrid_rag_cybersecurity.retrieval.vector_store import create_vector_store


# Run the workflow described in this module overview.
def main() -> None:

    project_root = Path(__file__).resolve().parents[1]

    raw_data_directory = project_root / "data" / "raw"
    vector_database_directory = project_root / "data" / "chroma_db"

    print("=" * 60)
    print("SENTINEL CYBERSECURITY - VECTOR DATABASE BUILDER")
    print("=" * 60)

    # Step 1: Load source documents.
    documents, failures = load_directory(raw_data_directory)

    print(f"\nExtracted records: {len(documents)}")

    # Step 2: Clean documents.
    cleaned_documents = [clean_text(document) for document in documents]

    print(f"Cleaned records: {len(cleaned_documents)}")

    # Step 3: Generate chunks.
    chunked_documents = chunk_documents(
        cleaned_documents,
        chunk_size=1200,
        chunk_overlap=200,
    )

    print(f"Generated chunks: {len(chunked_documents)}")

    # Step 4: Generate embeddings and store them.
    create_vector_store(
        documents=chunked_documents,
        persist_directory=vector_database_directory,
    )

    print("\nVECTOR DATABASE CREATED")
    print(f"Location: {vector_database_directory}")

    print(f"Ingestion failures: {len(failures)}")


if __name__ == "__main__":
    main()
