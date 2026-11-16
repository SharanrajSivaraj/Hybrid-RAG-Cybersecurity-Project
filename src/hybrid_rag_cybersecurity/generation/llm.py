"""Factory for the configured local Ollama chat model."""

from langchain_ollama import ChatOllama


def get_llm() -> ChatOllama:
    """
    Initialise and return the local Qwen3 language model.
    """

    return ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.1,
    )
