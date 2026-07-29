"""Hybrid retrieval using Reciprocal Rank Fusion (RRF)."""

from typing import Any

from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi

from src.bm25_retriever import search_bm25
from src.embeddings import generate_embeddings
from src.vector_store import search_documents


def reciprocal_rank_fusion(
    semantic_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Combine semantic and BM25 results using
    Reciprocal Rank Fusion (RRF).

    Args:
        semantic_results:
            Results from semantic search.

        bm25_results:
            Results from BM25 search.

        k:
            RRF constant.

    Returns:
        Combined ranked results.
    """
    scores: dict[str, float] = {}
    documents: dict[str, dict[str, Any]] = {}

    for rank, document in enumerate(semantic_results):
        doc_id = (
            f"{document['metadata']['source']}_"
            f"{document['metadata']['chunk_id']}"
        )

        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        documents[doc_id] = document

    for rank, document in enumerate(bm25_results):
        doc_id = (
            f"{document['metadata']['source']}_"
            f"{document['metadata']['chunk_id']}"
        )

        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        documents[doc_id] = document

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        documents[doc_id]
        for doc_id, _ in ranked
    ]


def hybrid_search(
    question: str,
    model,
    collection: Collection,
    bm25: BM25Okapi,
    documents: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Perform hybrid retrieval.

    Args:
        question:
            User question.

        model:
            SentenceTransformer model.

        collection:
            ChromaDB collection.

        bm25:
            BM25 index.

        documents:
            Chunked documents.

        top_k:
            Number of retrieved chunks.

    Returns:
        Hybrid ranked document chunks.
    """
    query_embedding = generate_embeddings(
        model,
        [question],
    )[0]

    semantic = search_documents(
        collection=collection,
        query_embedding=query_embedding,
        top_k=top_k,
    )

    semantic_results = []

    for document, metadata in zip(
        semantic["documents"][0],
        semantic["metadatas"][0],
    ):
        semantic_results.append(
            {
                "text": document,
                "metadata": metadata,
            }
        )

    bm25_results = search_bm25(
        bm25=bm25,
        documents=documents,
        query=question,
        top_k=top_k,
    )

    return reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
    )[:top_k]

if __name__ == "__main__":
    from src.document_loader import load_documents
    from src.text_chunker import chunk_documents
    from src.embeddings import (
        embed_documents,
        load_embedding_model,
    )
    from src.vector_store import (
        create_collection,
        add_documents,
    )
    from src.bm25_retriever import build_bm25

    print("Loading documents...")
    documents = load_documents("data/raw")

    print("Chunking...")
    chunks = chunk_documents(documents)

    print("Loading embedding model...")
    model = load_embedding_model()

    print("Generating embeddings...")
    embedded_documents = embed_documents(
        chunks,
        model,
    )

    print("Creating vector database...")
    collection = create_collection(reset=True)

    add_documents(
        collection,
        embedded_documents,
    )

    print("Building BM25...")
    bm25, indexed_chunks = build_bm25(chunks)

    question = "What is the VPN policy?"

    print(f"\nQuestion: {question}")

    results = hybrid_search(
        question=question,
        model=model,
        collection=collection,
        bm25=bm25,
        documents=indexed_chunks,
        top_k=3,
    )

    print("\nHybrid Retrieval Results\n")

    for index, result in enumerate(results, start=1):
        print(f"Result {index}")
        print(f"Source : {result['metadata']['source']}")
        print(f"Chunk  : {result['metadata']['chunk_id']}")
        print("-" * 70)
        print(result["text"][:300])
        print()