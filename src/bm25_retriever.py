"""BM25 keyword retrieval."""

import re
from typing import Any

from rank_bm25 import BM25Okapi


STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "of",
    "to",
    "for",
    "in",
    "on",
    "at",
    "by",
    "and",
    "or",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
}


def tokenize(text: str) -> list[str]:
    """
    Convert text into cleaned lowercase tokens.

    Args:
        text:
            Input text.

    Returns:
        List of cleaned tokens.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())

    return [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]


def build_bm25(
    documents: list[dict[str, Any]],
) -> tuple[BM25Okapi, list[dict[str, Any]]]:
    """
    Build a BM25 index.

    Args:
        documents:
            Chunked documents.

    Returns:
        BM25 model and original documents.
    """
    tokenized_corpus = [
        tokenize(doc["text"])
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, documents


def search_bm25(
    bm25: BM25Okapi,
    documents: list[dict[str, Any]],
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Search documents using BM25.

    Args:
        bm25:
            BM25 model.

        documents:
            Original chunked documents.

        query:
            User question.

        top_k:
            Number of results.

    Returns:
        Top matching document chunks.
    """
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    return [
        documents[index]
        for index in ranked_indices
    ]


if __name__ == "__main__":
    from src.document_loader import load_documents
    from src.text_chunker import chunk_documents

    print("Loading documents...")
    documents = load_documents("data/raw")

    print("Chunking...")
    chunks = chunk_documents(documents)

    print("Building BM25 index...")
    bm25, indexed_chunks = build_bm25(chunks)

    query = "What is the VPN policy?"

    print(f"\nQuestion: {query}")

    results = search_bm25(
        bm25=bm25,
        documents=indexed_chunks,
        query=query,
        top_k=3,
    )

    print("\nTop Results")
    print("=" * 70)

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Source : {result['metadata']['source']}")
        print(f"Chunk  : {result['metadata']['chunk_id']}")
        print("-" * 70)
        print(result["text"][:300])