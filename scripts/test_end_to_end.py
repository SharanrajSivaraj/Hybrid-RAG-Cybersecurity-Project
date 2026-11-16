"""Manual end-to-end smoke test for retrieval, reranking, and answer generation."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from hybrid_rag_cybersecurity.embeddings.embedding_model import (
    get_embedding_model,
)
from hybrid_rag_cybersecurity.generation.answer_generator import (
    generate_answer,
)
from hybrid_rag_cybersecurity.generation.context_builder import (
    build_context,
)
from hybrid_rag_cybersecurity.ingestion import load_directory
from hybrid_rag_cybersecurity.processing.chunker import chunk_documents
from hybrid_rag_cybersecurity.processing.text_cleaner import clean_text
from hybrid_rag_cybersecurity.reranking.reranker import (
    get_reranker,
)
from hybrid_rag_cybersecurity.retrieval.pipeline import (
    retrieve_and_rerank,
)
from hybrid_rag_cybersecurity.retrieval.tokenizer import tokenize_text


# Run the workflow described in this module overview.
def main():

    # --------------------------------------------------
    # 1. Project paths and question
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    raw_data_path = project_root / "data" / "raw"
    chroma_path = project_root / "data" / "chroma_db"

    question = "What are the NIST CSF 2.0 Respond and Recover functions about?"

    retrieval_queries = {
        "Respond": (
            "NIST CSF 2.0 RESPOND RS: "
            "Incident Management RS.MA, "
            "Incident Analysis RS.AN, "
            "Incident Response Reporting and Communication RS.CO, "
            "Incident Mitigation RS.MI. "
            "What cybersecurity outcomes are defined?"
        ),
        "Recover": (
            "NIST CSF 2.0 RECOVER RC: "
            "Incident Recovery Plan Execution RC.RP, "
            "Incident Recovery Communication RC.CO. "
            "What cybersecurity outcomes are defined?"
        ),
    }

    # --------------------------------------------------
    # 2. Load and process documents
    # --------------------------------------------------

    print("\n[1/5] Loading documents...")

    documents, failures = load_directory(raw_data_path)

    print(f"Extracted records: {len(documents)}")
    print(f"Loading failures: {len(failures)}")

    print("\n[2/5] Cleaning and chunking documents...")

    cleaned_documents = [clean_text(document) for document in documents]

    chunked_documents = chunk_documents(cleaned_documents)

    langchain_documents = [
        Document(
            page_content=document.text,
            metadata=document.metadata,
        )
        for document in chunked_documents
    ]

    print(f"Generated chunks: {len(langchain_documents)}")

    # --------------------------------------------------
    # 3. Initialise retrieval components
    # --------------------------------------------------

    print("\n[3/5] Initialising retrieval components...")

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name="cybersecurity_knowledge",
        embedding_function=embedding_model,
        persist_directory=str(chroma_path),
    )

    bm25_retriever = BM25Retriever.from_documents(
        langchain_documents,
        k=20,
        preprocess_func=tokenize_text,
    )

    reranker = get_reranker()

    # --------------------------------------------------
    # 4. Hybrid retrieval and reranking
    # --------------------------------------------------

    print("\n[4/5] Performing hybrid retrieval...")

    reranked_results = retrieve_and_rerank(
        question=question,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        retrieval_queries=retrieval_queries,
        semantic_k=20,
        rrf_top_n=30,
        rerank_top_n=4,
    )

    print(f"\nTotal unique retrieved documents: {len(reranked_results)}")

    for index, (document, score) in enumerate(
        reranked_results,
        start=1,
    ):
        print(f"\nResult {index}")
        print(f"Score: {score:.4f}")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Page: {document.metadata.get('page_number')}")
        print(f"Chunk: {document.metadata.get('chunk_index')}")
        print(f"Content:\n{document.page_content[:500]}")

    # --------------------------------------------------
    # 5. Generate grounded answer
    # --------------------------------------------------

    print("\n[5/5] Generating answer with Qwen3...")

    context = build_context(reranked_results)

    print("\n" + "=" * 70)
    print("FINAL GENERATION CONTEXT")
    print("=" * 70)
    print(context)
    print("=" * 70)

    answer = generate_answer(
        question=question,
        reranked_documents=reranked_results,
    )

    print("\n" + "=" * 65)
    print("SENTINEL CYBERSECURITY - HYBRID RAG")
    print("=" * 65)

    print(f"\nQuestion:\n{question}")
    print(f"\nAnswer:\n{answer}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
