"""Manual diagnostic for inspecting cybersecurity query tokenization."""

from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from hybrid_rag_cybersecurity.ingestion import load_directory
from hybrid_rag_cybersecurity.processing.chunker import chunk_documents
from hybrid_rag_cybersecurity.processing.text_cleaner import clean_text
from hybrid_rag_cybersecurity.retrieval.tokenizer import tokenize_text


# Run the workflow described in this module overview.
def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_data_path = project_root / "data" / "raw"

    documents, _failures = load_directory(raw_data_path)

    cleaned_documents = [clean_text(document) for document in documents]

    chunked_documents = chunk_documents(cleaned_documents)

    langchain_documents = [
        Document(
            page_content=document.text,
            metadata=document.metadata,
        )
        for document in chunked_documents
    ]

    bm25 = BM25Retriever.from_documents(
        langchain_documents,
        k=10,
    )

    query = "Which asset is associated with CVE-2025-1001?"

    print("\nBM25 TOKEN INSPECTION")
    print("=" * 65)

    print("Original query:")
    print(query)

    target_document = next(
        document
        for document in langchain_documents
        if document.metadata.get("source") == "vulnerability_register.xlsx"
        and document.metadata.get("sheet_name") == "Critical Vulnerabilities"
        and document.metadata.get("row_number") == 2
    )
    print("\nTokenized query:")
    print(tokenize_text(query))
    print(tokenize_text(target_document.page_content))

    print("\nTarget document:")
    print(target_document.page_content)

    print("\nTokenized target document:")
    print(bm25.preprocess_func(target_document.page_content))


if __name__ == "__main__":
    main()
