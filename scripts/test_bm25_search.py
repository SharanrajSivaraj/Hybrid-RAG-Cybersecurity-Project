"""Manual smoke test for keyword retrieval over project documents."""

from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from hybrid_rag_cybersecurity.ingestion import load_directory
from hybrid_rag_cybersecurity.processing.chunker import chunk_documents
from hybrid_rag_cybersecurity.processing.text_cleaner import clean_text


# Run the workflow described in this module overview.
def main() -> None:

    project_root = Path(__file__).resolve().parents[1]

    raw_data_directory = project_root / "data" / "raw"

    # Step 1: Load documents.
    documents, _failures = load_directory(raw_data_directory)

    # Step 2: Clean documents.
    cleaned_documents = [clean_text(document) for document in documents]

    # Step 3: Chunk documents.
    chunked_documents = chunk_documents(
        cleaned_documents,
        chunk_size=1200,
        chunk_overlap=200,
    )

    # Step 4: Convert to LangChain Document objects.
    langchain_documents = [
        Document(
            page_content=document.text,
            metadata=document.metadata,
        )
        for document in chunked_documents
    ]

    # Step 5: Build BM25 retriever.
    bm25_retriever = BM25Retriever.from_documents(
        langchain_documents,
        k=5,
    )

    query = "CVE-2025-1001"  # test with different queries
    # query = "Remote Code Execution"
    # query = "VPN Gateway"
    # query = "Authentication Bypass"

    print("=" * 60)
    print("SENTINEL CYBERSECURITY - BM25 SEARCH")
    print("=" * 60)

    print(f"\nQuery: {query}")

    results = bm25_retriever.invoke(query)

    print("\nTOP 5 RESULTS")
    print("-" * 60)

    for rank, document in enumerate(results, start=1):
        print(f"\nResult {rank}")

        print(f"Source: {document.metadata.get('source')}")

        print(f"File type: {document.metadata.get('file_type')}")

        print(f"Page: {document.metadata.get('page_number', 'N/A')}")

        print(f"Row: {document.metadata.get('row_number', 'N/A')}")

        print(f"\nContent:\n{document.page_content[:500]}")

    print("\nBM25 SEARCH COMPLETED")


if __name__ == "__main__":
    main()
