"""Manual smoke test for hybrid retrieval followed by cross-encoder reranking."""

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
from hybrid_rag_cybersecurity.reranking.reranker import (
    get_reranker,
    rerank_documents,
)
from hybrid_rag_cybersecurity.retrieval.hybrid_retriever import (
    reciprocal_rank_fusion,
)
from hybrid_rag_cybersecurity.retrieval.tokenizer import tokenize_text


# Run the workflow described in this module overview.
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_data_directory = project_root / "data" / "raw"
    vector_database_directory = project_root / "data" / "chroma_db"

    query = (
        "What steps should the SOC team follow "
        "when responding to a ransomware incident?"
    )

    # 1. Load and process documents
    documents, _failures = load_directory(raw_data_directory)

    cleaned_documents = [clean_text(document) for document in documents]

    chunked_documents = chunk_documents(
        cleaned_documents,
        chunk_size=1200,
        chunk_overlap=200,
    )

    langchain_documents = [
        Document(
            page_content=document.text,
            metadata=document.metadata,
        )
        for document in chunked_documents
    ]

    # 2. Initialise semantic retrieval
    vector_store = Chroma(
        collection_name="cybersecurity_knowledge",
        embedding_function=get_embedding_model(),
        persist_directory=str(vector_database_directory),
    )

    # 3. Initialise BM25 retrieval
    bm25_retriever = BM25Retriever.from_documents(
        langchain_documents,
        k=10,
        preprocess_func=tokenize_text,
    )

    # 4. Retrieve candidates from both methods
    semantic_results = vector_store.similarity_search(
        query,
        k=10,
    )

    keyword_results = bm25_retriever.invoke(query)

    # 5. Combine candidates using RRF
    hybrid_results = reciprocal_rank_fusion(
        ranked_lists=[
            semantic_results,
            keyword_results,
        ],
        k=60,
        top_n=20,
    )

    candidate_documents = [document for document, score in hybrid_results]

    # 6. Load cross-encoder
    reranker = get_reranker()

    # 7. Rerank hybrid candidates
    reranked_results = rerank_documents(
        query=query,
        documents=candidate_documents,
        reranker=reranker,
        top_n=5,
    )

    # 8. Display results
    print("=" * 60)
    print("SENTINEL CYBERSECURITY - RERANKER TEST")
    print("=" * 60)

    print(f"\nQuery: {query}")
    print(f"Semantic candidates: {len(semantic_results)}")
    print(f"BM25 candidates: {len(keyword_results)}")
    print(f"Hybrid candidates: {len(candidate_documents)}")

    print("\nRERANKED RESULTS")
    print("-" * 60)

    for rank, (document, score) in enumerate(
        reranked_results,
        start=1,
    ):
        print(f"\nResult {rank}")
        print(f"Score: {score:.4f}")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Page: {document.metadata.get('page_number', 'N/A')}")
        print(f"Row: {document.metadata.get('row_number', 'N/A')}")
        print(f"\nContent:\n{document.page_content[:500]}")

    print("\nRERANKER TEST COMPLETED")


if __name__ == "__main__":
    main()
