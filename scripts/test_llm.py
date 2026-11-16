"""Manual connectivity check for the configured local language model."""

from hybrid_rag_cybersecurity.generation.llm import get_llm


# Run the workflow described in this module overview.
def main():
    llm = get_llm()

    question = "Explain the role of a SOC analyst in three sentences."

    response = llm.invoke(question)

    print("\n" + "=" * 60)
    print("QWEN3 RESPONSE")
    print("=" * 60)
    print(response.content)


if __name__ == "__main__":
    main()
