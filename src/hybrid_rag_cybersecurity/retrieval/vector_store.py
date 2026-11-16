"""Chroma indexing workflow and deterministic chunk identifiers."""

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from hybrid_rag_cybersecurity.embeddings.embedding_model import (
    get_embedding_model,
)
from hybrid_rag_cybersecurity.ingestion.document_schema import (
    DocumentRecord,
)


def generate_document_id(document: DocumentRecord) -> str:
    """
    Generate a deterministic ID for a document chunk.
    """

    content = (
        document.metadata.get("source", "")
        + str(document.metadata.get("page_number", ""))
        + str(document.metadata.get("row_number", ""))
        + str(document.metadata.get("chunk_index", ""))
        + document.text
    )

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def create_vector_store(
    documents: list[DocumentRecord],
    persist_directory: str | Path,
) -> Chroma:
    """
    Convert DocumentRecords into LangChain Documents
    and store their embeddings in ChromaDB.
    """

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name="cybersecurity_knowledge",
        embedding_function=embedding_model,
        persist_directory=str(persist_directory),
    )

    batch_size = 64

    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]

        langchain_documents = [
            Document(
                page_content=document.text,
                metadata=document.metadata,
            )
            for document in batch
        ]

        ids = [generate_document_id(document) for document in batch]

        vector_store.add_documents(
            documents=langchain_documents,
            ids=ids,
        )

        print(
            f"Embedded and stored "
            f"{min(start + batch_size, len(documents))}"
            f"/{len(documents)} chunks"
        )

    return vector_store
