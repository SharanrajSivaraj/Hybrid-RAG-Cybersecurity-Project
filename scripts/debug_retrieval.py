"""Manual diagnostic for inspecting hybrid retrieval results."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from hybrid_rag_cybersecurity.embeddings.embedding_model import get_embedding_model
from hybrid_rag_cybersecurity.ingestion import load_directory
from hybrid_rag_cybersecurity.processing.chunker import chunk_documents
from hybrid_rag_cybersecurity.processing.text_cleaner import clean_text
from hybrid_rag_cybersecurity.retrieval.tokenizer import tokenize_text


# Run the workflow described in this module overview.
def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_data_path = project_root / "data" / "raw"
    chroma_path = project_root / "data" / "chroma_db"

    question = "Which asset is associated with CVE-2025-1001?"

    documents, _failures = load_directory(raw_data_path)
    cleaned_documents = [clean_text(doc) for doc in documents]
    chunked_documents = chunk_documents(cleaned_documents)

    langchain_documents = [
        Document(
            page_content=doc.text,
            metadata=doc.metadata,
        )
        for doc in chunked_documents
    ]

    # Inspect the original Excel records.
    print("\nEXCEL RECORDS")
    print("=" * 65)

    for doc in langchain_documents:
        if doc.metadata.get("source") == "vulnerability_register.xlsx":
            print("Metadata:", doc.metadata)
            print("Content:", doc.page_content)
            print("-" * 65)

    # Test BM25 retrieval.
    print("\nBM25 RESULTS")
    print("=" * 65)

    bm25_retriever = BM25Retriever.from_documents(
        langchain_documents,
        k=10,
        preprocess_func=tokenize_text,
    )

    for index, doc in enumerate(
        bm25_retriever.invoke(question),
        start=1,
    ):
        print(f"\nResult {index}")
        print("Source:", doc.metadata.get("source"))
        print("Sheet:", doc.metadata.get("sheet_name"))
        print("Row:", doc.metadata.get("row_number"))
        print("Content:", doc.page_content)

    # Test semantic retrieval.
    print("\nSEMANTIC RESULTS")
    print("=" * 65)

    vector_store = Chroma(
        collection_name="cybersecurity_knowledge",
        embedding_function=get_embedding_model(),
        persist_directory=str(chroma_path),
    )

    for index, doc in enumerate(
        vector_store.similarity_search(question, k=10),
        start=1,
    ):
        print(f"\nResult {index}")
        print("Source:", doc.metadata.get("source"))
        print("Sheet:", doc.metadata.get("sheet_name"))
        print("Row:", doc.metadata.get("row_number"))
        print("Content:", doc.page_content)


if __name__ == "__main__":
    main()
