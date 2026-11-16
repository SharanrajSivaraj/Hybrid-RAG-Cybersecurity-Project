"""Unit tests for the query service with model and vector-store boundaries mocked."""

from langchain_core.documents import Document

from hybrid_rag_cybersecurity.ingestion.document_schema import DocumentRecord
from hybrid_rag_cybersecurity.query_service import (
    answer_question,
    build_query_engine,
)


# Verify that build query engine mocks expensive components.
def test_build_query_engine_mocks_expensive_components(tmp_path, monkeypatch):
    raw_path = tmp_path / "data" / "raw"
    raw_path.mkdir(parents=True)
    chroma_path = tmp_path / "data" / "chroma_db"
    chroma_path.mkdir()
    (chroma_path / "chroma.sqlite3").touch()
    records = [
        DocumentRecord(
            "Incident response requires prompt triage.",
            {"source": "policy.txt", "file_type": "txt"},
        )
    ]

    # Fake collection proves setup does not need the real Chroma database.
    class FakeVectorStore:
        # Report a stored ID to satisfy the existing-collection check.
        def get(self, limit):
            assert limit == 1
            return {"ids": ["existing-id"]}

    # Fake keyword retriever avoids building an index in this unit test.
    class FakeBM25:
        # Confirm prepared chunks and return a lightweight retriever stand-in.
        @classmethod
        def from_documents(cls, documents, **kwargs):
            assert documents[0].page_content == records[0].text
            return cls()

    monkeypatch.setattr(
        "hybrid_rag_cybersecurity.query_service.load_directory",
        lambda path: (records, []),
    )
    monkeypatch.setattr(
        "hybrid_rag_cybersecurity.query_service.get_embedding_model",
        lambda: object(),
    )
    monkeypatch.setattr(
        "hybrid_rag_cybersecurity.query_service.Chroma",
        lambda **kwargs: FakeVectorStore(),
    )
    monkeypatch.setattr(
        "hybrid_rag_cybersecurity.query_service.BM25Retriever", FakeBM25
    )
    monkeypatch.setattr(
        "hybrid_rag_cybersecurity.query_service.get_reranker", lambda: object()
    )

    engine = build_query_engine(tmp_path)

    assert engine.vector_store is not None
    assert engine.bm25_retriever is not None
    assert engine.ingestion_failures == []


# Verify that answer question uses mocked retrieval and generation.
def test_answer_question_uses_mocked_retrieval_and_generation(monkeypatch):
    document = Document(
        page_content="Prompt triage has a target.",
        metadata={"source": "policy.txt", "file_type": "txt"},
    )
    engine = type(
        "Engine",
        (),
        {"vector_store": object(), "bm25_retriever": object(), "reranker": object()},
    )()
    monkeypatch.setattr(
        "hybrid_rag_cybersecurity.query_service.retrieve_and_rerank",
        lambda **kwargs: [(document, 0.9)],
    )
    monkeypatch.setattr(
        "hybrid_rag_cybersecurity.query_service.generate_answer",
        lambda **kwargs: "Prompt triage has a target. [S1]",
    )

    response = answer_question(engine, "What is the target?")

    assert response["answer"].endswith("[S1]")
    assert response["validation"]["valid"] is True
    assert response["results"][0][0] is document
