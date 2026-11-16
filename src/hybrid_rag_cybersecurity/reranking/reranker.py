"""Cross-encoder model setup and document score ordering."""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


def get_reranker() -> CrossEncoder:
    """
    Load the local cross-encoder reranking model.
    """

    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(
    query: str,
    documents: list[Document],
    reranker: CrossEncoder,
    top_n: int = 5,
) -> list[tuple[Document, float]]:
    """
    Rerank retrieved documents using a cross-encoder.
    """

    if not documents:
        return []

    pairs = [(query, document.page_content) for document in documents]

    scores = reranker.predict(pairs)

    scored_documents = list(zip(documents, scores, strict=True))

    ranked_documents = sorted(
        scored_documents,
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [(document, float(score)) for document, score in ranked_documents[:top_n]]
