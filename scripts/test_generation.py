"""Manual smoke test for local language-model answer generation."""

from langchain_core.documents import Document

from hybrid_rag_cybersecurity.generation.answer_generator import (
    generate_answer,
)


# Run the workflow described in this module overview.
def main():
    question = "What steps should the SOC team follow during incident response?"

    reranked_documents = [
        (
            Document(
                page_content=(
                    "The incident response process includes "
                    "detection, reporting, triage, containment, "
                    "eradication, recovery and lessons learned."
                ),
                metadata={
                    "source": "incident_response_policy.txt",
                    "file_type": "txt",
                },
            ),
            5.01,
        ),
        (
            Document(
                page_content=(
                    "The NIST Cybersecurity Framework includes "
                    "the Respond and Recover functions."
                ),
                metadata={
                    "source": "NIST.CSWP.29.pdf",
                    "file_type": "pdf",
                    "page_number": 9,
                },
            ),
            1.33,
        ),
    ]

    answer = generate_answer(
        question=question,
        reranked_documents=reranked_documents,
    )

    print("\n" + "=" * 60)
    print("SENTINEL CYBERSECURITY - GENERATED ANSWER")
    print("=" * 60)
    print(f"\nQuestion: {question}")
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
