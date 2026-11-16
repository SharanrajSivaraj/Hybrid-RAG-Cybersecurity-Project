"""Application-level setup for read-only RAG queries."""

from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from hybrid_rag_cybersecurity.embeddings.embedding_model import get_embedding_model
from hybrid_rag_cybersecurity.generation.answer_generator import generate_answer
from hybrid_rag_cybersecurity.generation.answer_validator import validate_answer
from hybrid_rag_cybersecurity.generation.context_builder import build_context
from hybrid_rag_cybersecurity.ingestion import load_directory
from hybrid_rag_cybersecurity.processing.chunker import chunk_documents
from hybrid_rag_cybersecurity.processing.text_cleaner import clean_text
from hybrid_rag_cybersecurity.reranking.reranker import get_reranker
from hybrid_rag_cybersecurity.retrieval.pipeline import retrieve_and_rerank
from hybrid_rag_cybersecurity.retrieval.tokenizer import tokenize_text


# Bundle the reusable components needed to answer a user question.
@dataclass
class QueryEngine:
    vector_store: Chroma
    bm25_retriever: BM25Retriever
    reranker: CrossEncoder
    ingestion_failures: list[dict[str, str]]


def build_query_engine(project_root: str | Path) -> QueryEngine:
    """Load query components without ingesting or writing to Chroma."""

    # Resolve the configured project root; callers cannot choose arbitrary paths.
    root = Path(project_root)
    raw_path = root / "data" / "raw"
    chroma_path = root / "data" / "chroma_db"
    if not (chroma_path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            f"Existing Chroma database not found at {chroma_path}. "
            "Build the vector store with the project script first."
        )

    records, failures = load_directory(raw_path)
    if not records:
        raise ValueError("No supported source documents were loaded from data/raw.")

    # Mirror the indexing pipeline so BM25 sees the same normalized chunks.
    cleaned = [clean_text(record) for record in records]
    chunks = chunk_documents(cleaned)
    documents = [
        Document(page_content=chunk.text, metadata=chunk.metadata) for chunk in chunks
    ]

    # Open the persistent collection for retrieval; indexing is a separate command.
    embedding_model = get_embedding_model()
    vector_store = Chroma(
        collection_name="cybersecurity_knowledge",
        embedding_function=embedding_model,
        persist_directory=str(chroma_path),
    )
    if not vector_store.get(limit=1).get("ids"):
        raise ValueError("The existing Chroma collection contains no documents.")

    bm25_retriever = BM25Retriever.from_documents(
        documents,
        k=20,
        preprocess_func=tokenize_text,
    )
    return QueryEngine(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        reranker=get_reranker(),
        ingestion_failures=failures,
    )


def answer_question(engine: QueryEngine, question: str) -> dict:
    # Retrieve evidence, generate a response, then validate its citations.
    """Retrieve evidence, generate an answer, and run structural checks."""

    results = retrieve_and_rerank(
        question=question,
        vector_store=engine.vector_store,
        bm25_retriever=engine.bm25_retriever,
        reranker=engine.reranker,
    )
    context = build_context(results)
    answer = generate_answer(question=question, reranked_documents=results)
    return {
        "answer": answer,
        "results": results,
        "validation": validate_answer(answer=answer, context=context),
    }
