"""Manual smoke test for hybrid semantic and keyword retrieval."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from hybrid_rag_cybersecurity.embeddings.embedding_model import (
    get_embedding_model,
)
from hybrid_rag_cybersecurity.ingestion import load_directory
from hybrid_rag_cybersecurity.processing.chunker import chunk_documents
from hybrid_rag_cybersecurity.processing.text_cleaner import clean_text
from hybrid_rag_cybersecurity.retrieval.hybrid_retriever import (
    reciprocal_rank_fusion,
)


# Run the workflow described in this module overview.
def main() -> None:

    project_root = Path(__file__).resolve().parents[1]

    raw_data_directory = project_root / "data" / "raw"
    vector_database_directory = project_root / "data" / "chroma_db"

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

    # Step 4: Convert to LangChain documents.
    langchain_documents = [
        Document(
            page_content=document.text,
            metadata=document.metadata,
        )
        for document in chunked_documents
    ]

    # Step 5: Initialise semantic retrieval.
    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name="cybersecurity_knowledge",
        embedding_function=embedding_model,
        persist_directory=str(vector_database_directory),
    )

    # Step 6: Initialise BM25 retrieval.
    bm25_retriever = BM25Retriever.from_documents(
        langchain_documents,
        k=10,
    )

    # Step 7: Execute query.
    # query = (
    #    "What steps should the SOC team follow "
    #    "when responding to a ransomware incident?"
    # )
    query = (
        "What steps should the SOC team follow when responding "
        "to a ransomware incident?"
    )

    print("=" * 60)
    print("SENTINEL CYBERSECURITY - HYBRID SEARCH")
    print("=" * 60)

    print(f"\nQuery: {query}")

    # Semantic results.
    semantic_results = vector_store.similarity_search(
        query,
        k=10,
    )

    # BM25 results.
    keyword_results = bm25_retriever.invoke(query)

    print("\nSEMANTIC RESULTS")
    print("-" * 60)

    for rank, document in enumerate(semantic_results, start=1):
        print(
            f"{rank}. {document.metadata.get('source')} | {document.page_content[:100]}"
        )

    print("\nBM25 RESULTS")
    print("-" * 60)

    for rank, document in enumerate(keyword_results, start=1):
        print(
            f"{rank}. {document.metadata.get('source')} | {document.page_content[:100]}"
        )

    # Step 8: Fuse results using RRF.
    hybrid_results = reciprocal_rank_fusion(
        ranked_lists=[
            semantic_results,
            keyword_results,
        ],
        k=60,
        top_n=10,
    )

    print("\nHYBRID RESULTS - RRF")
    print("-" * 60)

    for rank, (document, score) in enumerate(hybrid_results, start=1):
        print(f"\nResult {rank}")

        print(f"Source: {document.metadata.get('source')}")

        print(f"File type: {document.metadata.get('file_type')}")

        print(f"Page: {document.metadata.get('page_number', 'N/A')}")

        print(f"Row: {document.metadata.get('row_number', 'N/A')}")

        print(f"RRF score: {score:.6f}")

        print(f"Content: {document.page_content[:400]}")

    print("\nHYBRID SEARCH COMPLETED")


if __name__ == "__main__":
    main()
