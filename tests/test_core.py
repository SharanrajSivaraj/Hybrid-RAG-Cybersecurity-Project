"""Unit tests for retrieval, chunking, and citation validation."""

from langchain_core.documents import Document

from hybrid_rag_cybersecurity.generation.answer_validator import validate_answer
from hybrid_rag_cybersecurity.ingestion.document_schema import DocumentRecord
from hybrid_rag_cybersecurity.processing.chunker import chunk_documents
from hybrid_rag_cybersecurity.retrieval.hybrid_retriever import (
    reciprocal_rank_fusion,
)
from hybrid_rag_cybersecurity.retrieval.pipeline import retrieve_and_rerank


# Verify that chunker preserves source metadata and adds chunk metadata.
def test_chunker_preserves_source_metadata_and_adds_chunk_metadata():
    record = DocumentRecord(
        text="First paragraph. " * 12,
        metadata={"source": "policy.txt", "file_type": "txt"},
    )

    chunks = chunk_documents([record], chunk_size=40, chunk_overlap=5)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "policy.txt" for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )


# Verify that rrf deduplicates same document and returns scores.
def test_rrf_deduplicates_same_document_and_returns_scores():
    first = Document(page_content="same", metadata={"source": "a.txt"})
    second = Document(page_content="other", metadata={"source": "b.txt"})

    fused = reciprocal_rank_fusion([[first, second], [first]], top_n=5)

    assert [document.page_content for document, _ in fused] == ["same", "other"]
    assert fused[0][1] > fused[1][1]


# Verify that retrieval uses original question for empty query map.
def test_retrieval_uses_original_question_for_empty_query_map(monkeypatch):
    document = Document(page_content="evidence", metadata={"source": "policy.txt"})
    calls = []

    # Fake semantic retriever that returns a fixture and records each query.
    class VectorStore:
        # Record a semantic query and return the shared test document.
        def similarity_search(self, query, k):
            calls.append(("semantic", query))
            return [document]

    # Fake keyword retriever that returns a fixture and records each query.
    class BM25:
        # Record a keyword query and return the shared test document.
        def invoke(self, query):
            calls.append(("keyword", query))
            return [document]

    # Stand in for the cross-encoder and preserve candidate ordering.
    def rerank_documents(query, documents, reranker, top_n):
        calls.append(("rerank", query))
        return [(item, 1.0) for item in documents[:top_n]]

    monkeypatch.setattr(
        "hybrid_rag_cybersecurity.retrieval.pipeline.rerank_documents",
        rerank_documents,
    )

    results = retrieve_and_rerank(
        question="incident response",
        vector_store=VectorStore(),
        bm25_retriever=BM25(),
        reranker=object(),
        retrieval_queries={},
    )

    assert [kind for kind, _ in calls] == ["semantic", "keyword", "rerank"]
    assert all(query == "incident response" for _, query in calls)
    assert len(results) == 1


# Verify that validator reports unknown citation and uncited claim.
def test_validator_reports_unknown_citation_and_uncited_claim():
    validation = validate_answer(
        "This factual sentence is deliberately much longer than fifty characters "
        "[S2].\n"
        "Another significant factual statement is unsupported and has no citation.",
        "[S1]\nSource: policy.txt\nContent:\nEvidence.",
    )

    assert validation["valid"] is False
    assert validation["unknown_citations"] == ["S2"]
    assert len(validation["uncited_claim_lines"]) == 1
