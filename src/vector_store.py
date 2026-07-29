"""Vector database operations using ChromaDB."""

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection


def create_collection(
    collection_name: str = "enterprise_documents",
    reset: bool = False,
) -> Collection:
    """
    Create or retrieve a ChromaDB collection.

    Args:
        collection_name:
            Name of the collection.

        reset:
            If True, delete any existing collection before creating a new one.

    Returns:
        ChromaDB collection instance.
    """
    client = chromadb.PersistentClient(path="chroma_db")

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            # Collection may not exist yet.
            pass

    return client.get_or_create_collection(
        name=collection_name,
    )


def add_documents(
    collection: Collection,
    embedded_documents: list[dict[str, Any]],
) -> None:
    """
    Store embedded documents inside ChromaDB.

    Args:
        collection:
            ChromaDB collection.

        embedded_documents:
            Documents containing text, metadata,
            and embedding vectors.
    """
    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for index, document in enumerate(embedded_documents):
        ids.append(f"chunk_{index}")

        documents.append(document["text"])

        embeddings.append(document["embedding"])

        metadatas.append(document["metadata"])

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def search_documents(
    collection: Collection,
    query_embedding: list[float],
    top_k: int = 3,
) -> dict:
    """
    Search the vector database for similar document chunks.

    Args:
        collection:
            ChromaDB collection.

        query_embedding:
            Embedding vector of the user's query.

        top_k:
            Number of results to retrieve.

    Returns:
        Search results from ChromaDB.
    """
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )


if __name__ == "__main__":
    from src.document_loader import load_documents
    from src.text_chunker import chunk_documents
    from src.embeddings import (
        embed_documents,
        generate_embeddings,
        load_embedding_model,
    )

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

    print("Creating ChromaDB collection...")
    collection = create_collection(reset=True)

    print("Adding documents to ChromaDB...")
    add_documents(
        collection=collection,
        embedded_documents=embedded_documents,
    )

    print(f"\nDocuments stored: {collection.count()}")

    question = "How many vacation days do employees receive?"

    print("\nGenerating query embedding...")
    query_embedding = generate_embeddings(
        model=model,
        texts=[question],
    )[0]

    print("\nSearching ChromaDB...")
    results = search_documents(
        collection=collection,
        query_embedding=query_embedding,
        top_k=3,
    )

    print("\nQuestion:")
    print(question)

    print("\nTop Results:\n")

    for index, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][index]

        print(f"Result {index + 1}")
        print(f"Source : {metadata['source']}")
        print(f"Chunk  : {metadata['chunk_id']}")
        print()
        print(document[:250])
        print("-" * 70)