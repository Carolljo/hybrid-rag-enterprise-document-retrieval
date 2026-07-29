"""Utilities for structured source attribution in RAG responses."""

from typing import Any


def extract_citations(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Extract structured citation information from retrieved documents.

    Args:
        documents:
            Retrieved and reranked document chunks.

    Returns:
        List of unique citation dictionaries containing
        source file and chunk information.
    """
    citations = []
    seen = set()

    for document in documents:
        metadata = document.get("metadata", {})

        source = metadata.get("source", "unknown")
        chunk_id = metadata.get("chunk_id", "unknown")

        citation_key = (source, chunk_id)

        if citation_key in seen:
            continue

        seen.add(citation_key)

        citations.append(
            {
                "source": source,
                "chunk_id": chunk_id,
            }
        )

    return citations


def format_citations(
    citations: list[dict[str, Any]],
) -> list[str]:
    """
    Convert structured citations into readable labels.

    Args:
        citations:
            Structured citation dictionaries.

    Returns:
        Human-readable citation labels.
    """
    return [
        (
            f"{citation['source']} "
            f"(Chunk {citation['chunk_id']})"
        )
        for citation in citations
    ]


if __name__ == "__main__":
    test_documents = [
        {
            "text": "VPN policy information.",
            "metadata": {
                "source": "it_security_policy.txt",
                "chunk_id": 1,
            },
        },
        {
            "text": "Additional security information.",
            "metadata": {
                "source": "it_security_policy.txt",
                "chunk_id": 0,
            },
        },
        {
            "text": "Duplicate chunk.",
            "metadata": {
                "source": "it_security_policy.txt",
                "chunk_id": 1,
            },
        },
    ]

    citations = extract_citations(test_documents)

    print("Structured Citations:")
    print(citations)

    print("\nFormatted Citations:")

    for citation in format_citations(citations):
        print(f"- {citation}")