"""Grounded answer generation from retrieved document context."""

from langchain_core.documents import Document

from hybrid_rag_cybersecurity.generation.context_builder import (
    build_context,
)
from hybrid_rag_cybersecurity.generation.llm import get_llm
from hybrid_rag_cybersecurity.generation.prompt_template import (
    get_rag_prompt,
)


def generate_answer(
    question: str,
    reranked_documents: list[tuple[Document, float]],
) -> str:
    """
    Generate a grounded answer using reranked documents
    and the local Qwen3 model.
    """

    if not reranked_documents:
        return (
            "I could not find sufficient information in the "
            "available documents to answer this question."
        )

    # Step 1: Construct context from reranked documents.
    context = build_context(reranked_documents)

    # Step 2: Load the RAG prompt template.
    prompt = get_rag_prompt()

    # Step 3: Insert the question and retrieved context.
    formatted_prompt = prompt.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    # Step 4: Generate the answer using Qwen3.
    llm = get_llm()
    response = llm.invoke(formatted_prompt)

    return response.content
