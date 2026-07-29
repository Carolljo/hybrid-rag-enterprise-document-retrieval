"""Generate dense vector embeddings for document chunks."""

from typing import Any

from sentence_transformers import SentenceTransformer


def load_embedding_model(
    model_name: str = "all-MiniLM-L6-v2",
) -> SentenceTransformer:
    """
    Load the sentence-transformer embedding model.

    Args:
        model_name:
            Hugging Face model name.

    Returns:
        Loaded SentenceTransformer model.
    """
    return SentenceTransformer(model_name)


def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
) -> list[list[float]]:
    """
    Convert text into dense vector embeddings.

    Args:
        model:
            Loaded embedding model.

        texts:
            List of text chunks.

    Returns:
        List of embedding vectors.
    """
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
    )

    return embeddings.tolist()


def embed_documents(
    documents: list[dict[str, Any]],
    model: SentenceTransformer,
) -> list[dict[str, Any]]:
    """
    Attach embeddings to chunked documents.

    Args:
        documents:
            Chunked documents.

        model:
            Loaded embedding model.

    Returns:
        Documents enriched with embeddings.
    """
    texts = [
        document["text"]
        for document in documents
    ]

    vectors = generate_embeddings(
        model=model,
        texts=texts,
    )

    embedded_documents = []

    for document, vector in zip(
        documents,
        vectors,
    ):
        embedded_documents.append(
            {
                "text": document["text"],
                "metadata": document["metadata"],
                "embedding": vector,
            }
        )

    return embedded_documents
if __name__ == "__main__":
    from src.document_loader import load_documents
    from src.text_chunker import chunk_documents

    print("Loading documents...")
    documents = load_documents("data/raw")

    print("Chunking documents...")
    chunks = chunk_documents(documents)

    print("Loading embedding model...")
    model = load_embedding_model()

    print("Generating embeddings...")
    embedded_documents = embed_documents(
        documents=chunks,
        model=model,
    )

    print(f"\nTotal embedded chunks: {len(embedded_documents)}")

    first_embedding = embedded_documents[0]["embedding"]

    print(f"Embedding dimensions: {len(first_embedding)}")

    print("\nFirst chunk source:")
    print(embedded_documents[0]["metadata"]["source"])

    print("\nFirst five embedding values:")
    print(first_embedding[:5])