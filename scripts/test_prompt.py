"""Manual inspection of the project RAG prompt and local model response."""

from hybrid_rag_cybersecurity.generation.llm import get_llm
from hybrid_rag_cybersecurity.generation.prompt_template import get_rag_prompt


# Run the workflow described in this module overview.
def main():
    llm = get_llm()
    prompt = get_rag_prompt()

    question = "What are the steps in incident response?"

    context = """
    [S1]
    Source: incident_response_policy.txt

    Sentinel Cybersecurity follows a seven-step incident
    response process:

    1. Detection
    2. Reporting
    3. Triage
    4. Containment
    5. Eradication
    6. Recovery
    7. Lessons Learned
    """

    formatted_prompt = prompt.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    response = llm.invoke(formatted_prompt)

    print("\n" + "=" * 60)
    print("RAG PROMPT TEST")
    print("=" * 60)
    print(response.content)


if __name__ == "__main__":
    main()
