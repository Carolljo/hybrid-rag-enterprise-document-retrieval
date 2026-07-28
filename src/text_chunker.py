"""Utilities for splitting loaded documents into retrieval-ready chunks."""

from typing import Any

from src.document_loader import load_documents


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[str]:
    """
    Split document text into overlapping character-based chunks.

    Args:
        text: Document text that should be divided into chunks.
        chunk_size: Maximum number of characters in each chunk.
        chunk_overlap: Number of characters shared between adjacent chunks.

    Returns:
        A list containing the generated text chunks.

    Raises:
        ValueError: If chunk settings are invalid or the input text is empty.
    """
    if not text.strip():
        raise ValueError("Cannot chunk empty text.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - chunk_overlap

    return chunks


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict[str, Any]]:
    """
    Split loaded documents while preserving their source metadata.

    Args:
        documents: Documents returned by the document-loading stage.
        chunk_size: Maximum number of characters in each chunk.
        chunk_overlap: Number of characters shared between adjacent chunks.

    Returns:
        A list of dictionaries containing chunk text and metadata.

    Raises:
        ValueError: If the supplied document collection is empty.
    """
    if not documents:
        raise ValueError("No documents were provided for chunking.")

    chunked_documents = []

    for document in documents:
        text_chunks = chunk_text(
            text=document["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk in enumerate(text_chunks):
            metadata = document["metadata"].copy()
            metadata["chunk_id"] = chunk_index

            chunked_documents.append(
                {
                    "text": chunk,
                    "metadata": metadata,
                }
            )

    return chunked_documents


if __name__ == "__main__":
    loaded_documents = load_documents("data/raw")

    chunks = chunk_documents(
        loaded_documents,
        chunk_size=800,
        chunk_overlap=150,
    )

    print(f"Loaded documents: {len(loaded_documents)}")
    print(f"Generated chunks: {len(chunks)}\n")

    for chunk in chunks:
        print(
            f"Source: {chunk['metadata']['source']} | "
            f"Chunk: {chunk['metadata']['chunk_id']} | "
            f"Characters: {len(chunk['text'])}"
        )