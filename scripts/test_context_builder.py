"""Manual smoke test for formatting retrieved documents as cited context."""

from langchain_core.documents import Document

from hybrid_rag_cybersecurity.generation.context_builder import (
    build_context,
)


# Run the workflow described in this module overview.
def main():
    test_documents = [
        (
            Document(
                page_content=(
                    "The incident response process includes "
                    "detection, reporting, triage and containment."
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
        (
            Document(
                page_content=("CVE-2025-1001 is classified as Critical."),
                metadata={
                    "source": "vulnerability_register.xlsx",
                    "file_type": "xlsx",
                    "sheet_name": "Critical Vulnerabilities",
                    "row_number": 2,
                },
            ),
            1.19,
        ),
    ]

    context = build_context(test_documents)

    print("\n" + "=" * 60)
    print("CONTEXT BUILDER TEST")
    print("=" * 60)
    print(context)


if __name__ == "__main__":
    main()
