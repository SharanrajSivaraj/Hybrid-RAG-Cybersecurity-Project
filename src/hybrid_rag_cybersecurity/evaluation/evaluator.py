"""Evaluation workflow for retrieval quality and answer structure."""

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
from hybrid_rag_cybersecurity.generation.answer_validator import (
    validate_answer,
)
from hybrid_rag_cybersecurity.generation.context_builder import (
    build_context,
)
from hybrid_rag_cybersecurity.ingestion import load_directory
from hybrid_rag_cybersecurity.processing.chunker import chunk_documents
from hybrid_rag_cybersecurity.processing.text_cleaner import clean_text
from hybrid_rag_cybersecurity.reranking.reranker import get_reranker
from hybrid_rag_cybersecurity.retrieval.pipeline import (
    retrieve_and_rerank as run_retrieval_pipeline,
)
from hybrid_rag_cybersecurity.retrieval.tokenizer import tokenize_text


# Reuse one retrieval stack to score a collection of evaluation cases.
class RAGEvaluator:
    # Initialize the evaluator components and source corpus.
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.raw_data_path = project_root / "data" / "raw"
        self.chroma_path = project_root / "data" / "chroma_db"

        print("Loading evaluation components...")

        documents, failures = load_directory(self.raw_data_path)

        if failures:
            print(f"Warning: {len(failures)} ingestion failures")

        cleaned_documents = [clean_text(document) for document in documents]

        chunked_documents = chunk_documents(cleaned_documents)

        self.langchain_documents = [
            Document(
                page_content=document.text,
                metadata=document.metadata,
            )
            for document in chunked_documents
        ]

        self.embedding_model = get_embedding_model()

        self.vector_store = Chroma(
            collection_name="cybersecurity_knowledge",
            embedding_function=self.embedding_model,
            persist_directory=str(self.chroma_path),
        )

        self.bm25_retriever = BM25Retriever.from_documents(
            self.langchain_documents,
            k=20,
            preprocess_func=tokenize_text,
        )

        self.reranker = get_reranker()

        print("Evaluation components loaded successfully.")

    # Provide the retrieve and rerank behavior used by this module.
    def retrieve_and_rerank(
        self,
        question: str,
        retrieval_queries: dict[str, str] | None = None,
    ) -> list[tuple[Document, float]]:

        return run_retrieval_pipeline(
            question=question,
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
            reranker=self.reranker,
            retrieval_queries=retrieval_queries,
            semantic_k=20,
            rrf_top_n=30,
            rerank_top_n=4,
        )

    # Provide the evaluate question behavior used by this module.
    def evaluate_question(self, test_case: dict) -> dict:

        question = test_case["question"]

        reranked_results = self.retrieve_and_rerank(
            question=question,
            retrieval_queries=test_case.get("retrieval_queries"),
        )

        context = build_context(reranked_results)

        answer = generate_answer(
            question=question,
            reranked_documents=reranked_results,
        )

        # Validate the generated answer against retrieved evidence.
        validation = validate_answer(
            answer=answer,
            context=context,
        )

        retrieved_sources = list(
            dict.fromkeys(
                document.metadata.get("source", "Unknown")
                for document, score in reranked_results
            )
        )

        expected_sources = test_case["expected_sources"]
        expected_keywords = test_case["expected_keywords"]

        answer_lower = answer.lower()

        matched_keywords = [
            keyword for keyword in expected_keywords if keyword.lower() in answer_lower
        ]

        keyword_coverage = (
            len(matched_keywords) / len(expected_keywords)
            if expected_keywords
            else None
        )

        source_coverage = (
            len(set(expected_sources) & set(retrieved_sources)) / len(expected_sources)
            if expected_sources
            else None
        )

        return {
            "id": test_case["id"],
            "question": question,
            "answer": answer,
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
            "matched_keywords": matched_keywords,
            "keyword_coverage": keyword_coverage,
            "source_coverage": source_coverage,
            "answerable": test_case["answerable"],
            "validation": validation,
        }
