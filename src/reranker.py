"""Rerank retrieved document chunks using a CrossEncoder model."""

from typing import Any

from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> CrossEncoder:
    """
    Load the CrossEncoder reranking model.

    Args:
        model_name:
            Hugging Face model used for reranking.

    Returns:
        Loaded CrossEncoder model.
    """
    return CrossEncoder(model_name)


def rerank_documents(
    query: str,
    documents: list[dict[str, Any]],
    model: CrossEncoder,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Rerank retrieved documents according to their
    relevance to the user query.

    Args:
        query:
            User question.

        documents:
            Candidate document chunks returned by retrieval.

        model:
            Loaded CrossEncoder reranking model.

        top_k:
            Number of final results to return.

    Returns:
        Reranked document chunks with relevance scores.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not documents:
        return []

    pairs = [
        [query, document["text"]]
        for document in documents
    ]

    scores = model.predict(pairs)

    scored_documents = []

    for document, score in zip(documents, scores):
        reranked_document = {
            **document,
            "rerank_score": float(score),
        }

        scored_documents.append(reranked_document)

    scored_documents.sort(
        key=lambda document: document["rerank_score"],
        reverse=True,
    )

    return scored_documents[:top_k]


if __name__ == "__main__":
    from src.bm25_retriever import build_bm25
    from src.document_loader import load_documents
    from src.embeddings import (
        embed_documents,
        load_embedding_model,
    )
    from src.hybrid_retriever import hybrid_search
    from src.text_chunker import chunk_documents
    from src.vector_store import (
        add_documents,
        create_collection,
    )

    print("Loading documents...")
    documents = load_documents("data/raw")

    print("Chunking documents...")
    chunks = chunk_documents(documents)

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Generating embeddings...")
    embedded_documents = embed_documents(
        chunks,
        embedding_model,
    )

    print("Creating vector database...")
    collection = create_collection(reset=True)

    add_documents(
        collection,
        embedded_documents,
    )

    print("Building BM25 index...")
    bm25, indexed_chunks = build_bm25(chunks)

    question = "What is the VPN policy?"

    print("\nRunning hybrid retrieval...")

    # Retrieve more candidates than we ultimately need.
    candidate_results = hybrid_search(
        question=question,
        model=embedding_model,
        collection=collection,
        bm25=bm25,
        documents=indexed_chunks,
        top_k=5,
    )

    print("Loading reranker...")
    reranker = load_reranker()

    print("Reranking results...")

    results = rerank_documents(
        query=question,
        documents=candidate_results,
        model=reranker,
        top_k=3,
    )

    print(f"\nQuestion: {question}")
    print("\nReranked Results")
    print("=" * 70)

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print(f"Source : {result['metadata']['source']}")
        print(f"Chunk  : {result['metadata']['chunk_id']}")
        print(f"Score  : {result['rerank_score']:.4f}")
        print("-" * 70)
        print(result["text"][:300])