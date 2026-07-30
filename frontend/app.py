"""Streamlit frontend for the Hybrid RAG application."""

from typing import Any

import requests
import streamlit as st
import os


API_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000/ask",
)

def query_api(question: str) -> dict[str, Any]:
    """
    Send a user question to the FastAPI RAG backend.

    Args:
        question: User question submitted from the Streamlit interface.

    Returns:
        JSON response containing the generated answer,
        verified citations, and citation confidence.

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
    Render unique verified source citations returned by the RAG backend.

    Args:
        citations: List of citation dictionaries.
    """
    if not citations:
        st.info("No citations were returned.")
        return

    st.subheader("Sources")

    seen_sources = set()

    for citation in citations:
        source = citation.get(
            "source",
            "Unknown source",
        )

        if source in seen_sources:
            continue

        seen_sources.add(source)

        verified = citation.get(
            "verified",
            False,
        )

        verification_label = (
            "Verified"
            if verified
            else "Unverified"
        )

        st.write(
            f"- {source} — {verification_label}"
        )


def render_confidence(confidence: float) -> None:
    """
    Display citation provenance confidence.

    Args:
        confidence:
            Proportion of citations verified against
            retrieved document chunks.
    """
    confidence_percentage = confidence * 100

    st.metric(
        "Citation Verification",
        f"{confidence_percentage:.0f}%",
    )

    st.caption(
        "Measures whether returned citations correspond "
        "to retrieved document chunks. It does not measure "
        "overall answer correctness."
    )


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
        "and receive grounded answers with verified citations."
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
                result = query_api(
                    question.strip()
                )

            answer = result.get(
                "answer",
                "No answer was returned.",
            )

            citations = result.get(
                "citations",
                [],
            )

            citation_confidence = result.get(
                "citation_confidence",
                0.0,
            )

            st.subheader("Answer")
            st.write(answer)

            render_confidence(
                citation_confidence
            )

            render_citations(
                citations
            )

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