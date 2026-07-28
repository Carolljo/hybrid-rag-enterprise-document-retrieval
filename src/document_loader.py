"""Utilities for loading enterprise documents into the RAG pipeline."""

from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".txt"}


def load_text_document(file_path: Path) -> dict[str, Any]:
    """
    Load a text document and attach source metadata.

    Args:
        file_path: Path to the text document that should be loaded.

    Returns:
        A dictionary containing the document text and source metadata.

    Raises:
        FileNotFoundError: If the specified document does not exist.
        ValueError: If the document is empty.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    text = file_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Document is empty: {file_path}")

    return {
        "text": text,
        "metadata": {
            "source": file_path.name,
            "file_type": file_path.suffix.lower(),
        },
    }


def load_documents(directory: str | Path) -> list[dict[str, Any]]:
    """
    Load all supported enterprise documents from a directory.

    Args:
        directory: Directory containing documents to ingest.

    Returns:
        A list of dictionaries containing document text and metadata.

    Raises:
        FileNotFoundError: If the supplied directory does not exist.
        NotADirectoryError: If the supplied path is not a directory.
        ValueError: If no supported documents are found.
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Document directory not found: {directory_path}"
        )

    if not directory_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {directory_path}"
        )

    documents = []

    for file_path in sorted(directory_path.iterdir()):
        if (
            file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            documents.append(load_text_document(file_path))

    if not documents:
        raise ValueError(
            f"No supported documents found in: {directory_path}"
        )

    return documents


if __name__ == "__main__":
    loaded_documents = load_documents("data/raw")

    print(f"Loaded {len(loaded_documents)} documents.\n")

    for document in loaded_documents:
        print(f"Source: {document['metadata']['source']}")
        print(f"Characters: {len(document['text'])}")
        print("-" * 50)