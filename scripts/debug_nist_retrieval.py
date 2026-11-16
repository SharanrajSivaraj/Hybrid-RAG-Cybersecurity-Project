"""Manual diagnostic for NIST-focused hybrid retrieval and reranking."""

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


def print_results(
    title: str,
    results: list[Document],
    limit: int = 1000,
) -> None:
    """Print retrieved documents and their metadata."""

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    for index, document in enumerate(results, start=1):
        metadata = document.metadata

        print(f"\nRESULT {index}")
        print("-" * 70)
        print("Source:", metadata.get("source"))
        print("Page:", metadata.get("page_number"))
        print("Chunk:", metadata.get("chunk_index"))
        print("Content:")
        print(document.page_content[:limit])


def print_reranked_results(
    results: list[tuple[Document, float]],
) -> None:
    """Print final reranked documents with cross-encoder scores."""

    print("\n" + "=" * 70)
    print("FINAL RERANKED RESULTS")
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    for index, (document, score) in enumerate(results, start=1):
        metadata = document.metadata

        print(f"\nRESULT {index}")
        print("-" * 70)
        print("Score:", round(score, 4))
        print("Source:", metadata.get("source"))
        print("Page:", metadata.get("page_number"))
        print("Chunk:", metadata.get("chunk_index"))
        print("Content:")
        print(document.page_content)


# Run the workflow described in this module overview.
def main():
    # --------------------------------------------------
    # 1. Project paths
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    raw_data_path = project_root / "data" / "raw"
    chroma_path = project_root / "data" / "chroma_db"

    # --------------------------------------------------
    # 2. Test queries
    # --------------------------------------------------

    questions = [
        "What is the purpose of the NIST CSF 2.0 Respond function?",
        "What cybersecurity outcomes does the Respond function define?",
        "Incident Management, Incident Analysis, Incident "
        "Response Reporting and Incident Mitigation under "
        "NIST CSF 2.0",
    ]

    # --------------------------------------------------
    # 3. Load and process documents
    # --------------------------------------------------

    print("\nLoading documents...")

    documents, failures = load_directory(raw_data_path)

    cleaned_documents = [clean_text(document) for document in documents]

    chunked_documents = chunk_documents(cleaned_documents)

    langchain_documents = [
        Document(
            page_content=document.text,
            metadata=document.metadata,
        )
        for document in chunked_documents
    ]

    print("Extracted records:", len(documents))
    print("Generated chunks:", len(chunked_documents))
    print("Ingestion failures:", len(failures))

    # --------------------------------------------------
    # 4. Initialise semantic retrieval
    # --------------------------------------------------

    print("\nInitialising semantic retrieval...")

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name="cybersecurity_knowledge",
        embedding_function=embedding_model,
        persist_directory=str(chroma_path),
    )

    # --------------------------------------------------
    # 5. Initialise BM25 retrieval
    # --------------------------------------------------

    print("Initialising BM25 retrieval...")

    bm25_retriever = BM25Retriever.from_documents(
        langchain_documents,
        k=20,
        preprocess_func=tokenize_text,
    )

    # --------------------------------------------------
    # 6. Initialise reranker
    # --------------------------------------------------

    print("Loading cross-encoder reranker...")

    reranker = get_reranker()

    # --------------------------------------------------
    # 7. Run experiments
    # --------------------------------------------------

    for query_number, question in enumerate(questions, start=1):
        print("\n\n")
        print("#" * 70)
        print(f"QUERY {query_number}")
        print("#" * 70)
        print("\nQuestion:", question)

        # ----------------------------------------------
        # A. Semantic retrieval
        # ----------------------------------------------

        semantic_results = vector_store.similarity_search(
            question,
            k=20,
        )

        print_results(
            title="SEMANTIC RETRIEVAL",
            results=semantic_results,
        )

        # ----------------------------------------------
        # B. BM25 retrieval
        # ----------------------------------------------

        bm25_results = bm25_retriever.invoke(question)

        print_results(
            title="BM25 RETRIEVAL",
            results=bm25_results,
        )

        # ----------------------------------------------
        # C. Reciprocal Rank Fusion
        # ----------------------------------------------

        fused_results = reciprocal_rank_fusion(
            ranked_lists=[
                semantic_results,
                bm25_results,
            ],
            k=60,
            top_n=30,
        )

        fused_documents = [document for document, score in fused_results]

        print_results(
            title="RRF FUSED RESULTS",
            results=fused_documents,
        )

        # ----------------------------------------------
        # D. Cross-encoder reranking
        # ----------------------------------------------

        reranked_results = rerank_documents(
            query=question,
            documents=fused_documents,
            reranker=reranker,
            top_n=10,
        )

        print_reranked_results(reranked_results)


if __name__ == "__main__":
    main()
