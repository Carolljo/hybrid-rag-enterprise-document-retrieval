"""Streamlit frontend for the Hybrid RAG application."""

from typing import Any

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/ask"


def query_api(question: str) -> dict[str, Any]:
    """
    Send a user question to the FastAPI RAG backend.

    Args:
        question: User question submitted from the Streamlit interface.

    Returns:
        JSON response containing the generated answer and citations.

    Raises:
        requests.RequestException: If communication with the API fails.
    """
    response = requests.post(
        API_URL,
        json={"question": question},
        timeout=120,
    )
    response.raise_for_status()

    return response.json()


def render_citations(citations: list[dict[str, Any]]) -> None:
    """
    Render unique source citations returned by the RAG backend.

    Args:
        citations: List of citation dictionaries.
    """
    if not citations:
        st.info("No citations were returned.")
        return

    st.subheader("Sources")

    seen_sources = set()

    for citation in citations:
        source = citation.get("source", "Unknown source")

        if source in seen_sources:
            continue

        seen_sources.add(source)
        st.write(f"- {source}")


def main() -> None:
    """Run the Streamlit frontend application."""
    st.set_page_config(
        page_title="Hybrid RAG",
        page_icon="🔎",
        layout="centered",
    )

    st.title("Hybrid RAG")
    st.write(
        "Ask questions about the indexed enterprise documents "
        "and receive answers with source citations."
    )

    question = st.text_area(
        "Question",
        placeholder="What is the remote work policy?",
        height=120,
    )

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        try:
            with st.spinner(
                "Searching documents and generating answer..."
            ):
                result = query_api(question.strip())

            answer = result.get(
                "answer",
                "No answer was returned.",
            )
            citations = result.get("citations", [])

            st.subheader("Answer")
            st.write(answer)

            render_citations(citations)

        except requests.RequestException as exc:
            st.error(
                f"Unable to communicate with the API: {exc}"
            )

        except (TypeError, ValueError, KeyError) as exc:
            st.error(
                f"Invalid response received from the API: {exc}"
            )


if __name__ == "__main__":
    main()