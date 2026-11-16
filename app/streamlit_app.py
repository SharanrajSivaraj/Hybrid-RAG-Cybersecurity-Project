"""Streamlit interface for asking questions over the cybersecurity knowledge base."""

import logging
from pathlib import Path

import streamlit as st

from hybrid_rag_cybersecurity.query_service import (
    answer_question,
    build_query_engine,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Load and return the cached search resources for Streamlit sessions.
@st.cache_resource(show_spinner="Loading the local search components…")
def get_query_engine():
    return build_query_engine(PROJECT_ROOT)


# Render the app page and process each submitted question.
def main() -> None:
    st.set_page_config(page_title="Sentinel Cybersecurity Assistant", page_icon="🛡️")
    st.title("🛡️ Sentinel Cybersecurity Assistant")
    st.caption(
        "Ask questions about the indexed security documents. "
        "Answers are generated locally with Ollama and include source references."
    )

    with st.sidebar:
        st.subheader("System status")
        st.write("Vector database: existing local collection")
        st.write("Language model: Ollama · qwen3:8b")
        st.info("Retrieved documents are treated as evidence, not instructions.")

    try:
        engine = get_query_engine()
    except Exception:
        # Keep local paths and dependency details out of the browser response.
        logger.exception("Search component initialization failed.")
        st.error("Search components could not be loaded. Check the local app setup.")
        st.caption(
            "Check that data/raw and the existing data/chroma_db are available, "
            "and that Ollama is running with the required models."
        )
        st.stop()

    if engine.ingestion_failures:
        st.warning(
            f"{len(engine.ingestion_failures)} source file(s) could not be read "
            "for keyword search. Semantic search remains available."
        )

    with st.form("question_form"):
        question = st.text_area(
            "Your question",
            placeholder="For example: What are the incident response steps?",
            max_chars=2000,
            height=100,
        )
        submitted = st.form_submit_button("Search documents", type="primary")

    if submitted:
        cleaned_question = question.strip()
        if not cleaned_question:
            st.info("Enter a question to search the documents.")
            return

        try:
            with st.spinner("Searching sources and preparing an answer…"):
                response = answer_question(engine, cleaned_question)
        except Exception:
            # Do not show raw exception details, which may expose local system paths.
            logger.exception("Question processing failed.")
            st.error("The question could not be answered. Check the local model setup.")
            st.caption("Check that Ollama is running and its local models are ready.")
            return

        st.subheader("Answer")
        # Streamlit uses safe Markdown rendering by default; raw HTML is not enabled.
        st.markdown(response["answer"])

        validation = response["validation"]
        if validation["unknown_citations"] or validation["unsupported_identifiers"]:
            st.warning(
                "The answer contains citation labels or NIST identifiers that "
                "could not be matched to the retrieved context. Review the "
                "answer and sources before relying on it."
            )
        if validation["uncited_claim_lines"]:
            st.info("Some answer lines may need a source citation.")

        st.subheader("Retrieved sources")
        if not response["results"]:
            st.write("No matching sources were retrieved.")
        for index, (document, score) in enumerate(response["results"], start=1):
            metadata = document.metadata
            details = [metadata.get("source", "Unknown source")]
            for label, key in (
                ("Page", "page_number"),
                ("Sheet", "sheet_name"),
                ("Row", "row_number"),
            ):
                value = metadata.get(key)
                if value is not None:
                    details.append(f"{label} {value}")
            with st.expander(f"[S{index}] {' · '.join(details)}"):
                st.caption(f"Reranker score: {score:.3f}")
                st.write(document.page_content)


if __name__ == "__main__":
    main()
