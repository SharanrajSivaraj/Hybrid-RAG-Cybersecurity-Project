"""Formatting retrieved passages with stable citation labels."""

from langchain_core.documents import Document


def build_context(
    reranked_documents: list[tuple[Document, float]],
) -> str:
    """
    Convert reranked documents into structured context
    with source references for the LLM.
    """

    context_parts = []

    for index, (document, _score) in enumerate(
        reranked_documents,
        start=1,
    ):
        metadata = document.metadata

        source = metadata.get("source", "Unknown source")
        file_type = metadata.get("file_type", "Unknown")

        citation = f"S{index}"

        source_details = [
            f"Source: {source}",
            f"File type: {file_type}",
        ]

        if metadata.get("page_number") is not None:
            source_details.append(f"Page: {metadata['page_number']}")

        if metadata.get("sheet_name"):
            source_details.append(f"Sheet: {metadata['sheet_name']}")

        if metadata.get("row_number") is not None:
            source_details.append(f"Row: {metadata['row_number']}")

        section = (
            f"[{citation}]\n"
            f"{' | '.join(source_details)}\n"
            f"Content:\n{document.page_content}"
        )

        context_parts.append(section)

    return "\n\n".join(context_parts)
