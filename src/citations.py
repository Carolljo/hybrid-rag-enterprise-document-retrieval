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


def verify_citations(
    citations: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Verify that citations reference retrieved document chunks.

    Args:
        citations:
            Structured citations extracted from retrieved documents.

        documents:
            Retrieved and reranked document chunks used as context.

    Returns:
        Citation dictionaries containing verification status.
    """
    available_chunks = {
        (
            document.get("metadata", {}).get("source"),
            document.get("metadata", {}).get("chunk_id"),
        )
        for document in documents
    }

    verified_citations = []

    for citation in citations:
        citation_key = (
            citation.get("source"),
            citation.get("chunk_id"),
        )

        verified_citations.append(
            {
                **citation,
                "verified": citation_key in available_chunks,
            }
        )

    return verified_citations


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

    verified_citations = verify_citations(
        citations,
        test_documents,
    )

    print("Structured Citations:")
    print(citations)

    print("\nVerified Citations:")
    print(verified_citations)

    print("\nFormatted Citations:")

    for citation in format_citations(verified_citations):
        print(f"- {citation}")