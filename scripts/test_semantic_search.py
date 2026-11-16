"""Manual smoke test for Chroma semantic search."""

from pathlib import Path

from langchain_chroma import Chroma

from hybrid_rag_cybersecurity.embeddings.embedding_model import (
    get_embedding_model,
)


# Run the workflow described in this module overview.
def main() -> None:

    project_root = Path(__file__).resolve().parents[1]

    vector_database_directory = project_root / "data" / "chroma_db"

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name="cybersecurity_knowledge",
        embedding_function=embedding_model,
        persist_directory=str(vector_database_directory),
    )

    query = (
        "What steps should the SOC team follow "
        "when responding to a ransomware incident?"
    )

    print("=" * 60)
    print("SENTINEL CYBERSECURITY - SEMANTIC SEARCH")
    print("=" * 60)

    print(f"\nQuery: {query}")

    results = vector_store.similarity_search_with_score(
        query,
        k=5,
    )

    print("\nTOP 5 RESULTS")
    print("-" * 60)

    for rank, (document, distance) in enumerate(results, start=1):
        print(f"\nResult {rank}")

        print(f"Source: {document.metadata.get('source')}")

        print(f"File type: {document.metadata.get('file_type')}")

        print(f"Page: {document.metadata.get('page_number', 'N/A')}")

        print(f"Row: {document.metadata.get('row_number', 'N/A')}")

        print(f"Distance: {distance}")

        print(f"\nContent:\n{document.page_content[:500]}")

    print("\nSEARCH COMPLETED")


if __name__ == "__main__":
    main()
