"""Text splitting that preserves source metadata and adds chunk metadata."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybrid_rag_cybersecurity.ingestion.document_schema import (
    DocumentRecord,
)


def chunk_documents(
    documents: list[DocumentRecord],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[DocumentRecord]:
    """
    Split documents using LangChain's recursive character splitter.

    Original metadata is preserved, with chunk-specific
    metadata added to each resulting record.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        add_start_index=True,
    )

    chunked_documents = []

    for document in documents:
        chunks = splitter.create_documents(
            texts=[document.text],
            metadatas=[document.metadata],
        )

        for chunk_index, chunk in enumerate(chunks):
            metadata = chunk.metadata.copy()

            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.page_content),
                }
            )

            chunked_documents.append(
                DocumentRecord(
                    text=chunk.page_content,
                    metadata=metadata,
                )
            )

    return chunked_documents
