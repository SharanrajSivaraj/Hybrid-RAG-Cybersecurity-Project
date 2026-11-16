"""Semantic, keyword, fusion, and reranking query workflow."""

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from hybrid_rag_cybersecurity.reranking.reranker import rerank_documents
from hybrid_rag_cybersecurity.retrieval.hybrid_retriever import (
    get_document_key,
    reciprocal_rank_fusion,
)


# Search both retrievers, fuse ranks, rerank candidates, and deduplicate.
def retrieve_and_rerank(
    question: str,
    vector_store: Chroma,
    bm25_retriever: BM25Retriever,
    reranker: CrossEncoder,
    retrieval_queries: dict[str, str] | None = None,
    semantic_k: int = 20,
    rrf_top_n: int = 30,
    rerank_top_n: int = 4,
) -> list[tuple[Document, float]]:

    # Use the original question if no decomposed queries are provided.
    if not retrieval_queries:
        retrieval_queries = {
            "main": question,
        }

    queries = [query.strip() for query in retrieval_queries.values() if query.strip()]
    if not queries:
        queries = [question]

    combined_results = []
    seen_documents = set()

    # Run the retrieval pipeline for each query.
    for retrieval_query in queries:
        # 1. Semantic retrieval
        semantic_results = vector_store.similarity_search(
            retrieval_query,
            k=semantic_k,
        )

        # 2. Keyword retrieval
        bm25_results = bm25_retriever.invoke(retrieval_query)

        # 3. Reciprocal Rank Fusion
        fused_results = reciprocal_rank_fusion(
            ranked_lists=[
                semantic_results,
                bm25_results,
            ],
            k=60,
            top_n=rrf_top_n,
        )

        # 4. Cross-encoder reranking
        reranked_query_results = rerank_documents(
            query=retrieval_query,
            documents=[document for document, score in fused_results],
            reranker=reranker,
            top_n=rerank_top_n,
        )

        # 5. Combine results and remove duplicate documents.
        for document, score in reranked_query_results:
            document_key = get_document_key(document)

            if document_key not in seen_documents:
                seen_documents.add(document_key)

                combined_results.append((document, score))

    return combined_results
