"""Compare retrieval performance across chunking configurations."""

import json
from pathlib import Path
from typing import Any

from src.bm25_retriever import build_bm25
from src.document_loader import load_documents
from src.embeddings import embed_documents, load_embedding_model
from src.hybrid_retriever import hybrid_search
from src.reranker import load_reranker, rerank_documents
from src.text_chunker import chunk_documents
from src.vector_store import add_documents, create_collection


EVALUATION_FILE = Path("evaluation/evaluation_questions.json")
DATA_DIRECTORY = "data/raw"

CHUNK_CONFIGURATIONS = [
    {
        "name": "small",
        "chunk_size": 400,
        "chunk_overlap": 75,
    },
    {
        "name": "current",
        "chunk_size": 800,
        "chunk_overlap": 150,
    },
    {
        "name": "large",
        "chunk_size": 1200,
        "chunk_overlap": 200,
    },
]


def load_questions() -> list[dict[str, Any]]:
    """Load answerable golden evaluation questions."""
    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    return [
        item
        for item in questions
        if item["answerable"]
    ]


def get_document_source(
    document: dict[str, Any],
) -> str | None:
    """Extract source filename from a retrieved document."""
    source = document.get("source")

    if source:
        return str(source)

    metadata = document.get("metadata", {})

    if isinstance(metadata, dict):
        source = metadata.get("source")

        if source:
            return str(source)

    return None


def evaluate_configuration(
    configuration: dict[str, Any],
    questions: list[dict[str, Any]],
    raw_documents: list[dict[str, Any]],
    embedding_model: Any,
    reranker: Any,
) -> dict[str, Any]:
    """Evaluate one chunking configuration."""

    chunks = chunk_documents(
        raw_documents,
        chunk_size=configuration["chunk_size"],
        chunk_overlap=configuration["chunk_overlap"],
    )

    embedded_documents = embed_documents(
        chunks,
        embedding_model,
    )

    collection = create_collection(reset=True)

    add_documents(
        collection,
        embedded_documents,
    )

    bm25, indexed_documents = build_bm25(chunks)

    hit_count = 0
    total_expected_sources = 0
    total_matched_sources = 0

    for item in questions:
        expected_sources = set(
            item["expected_sources"]
        )

        candidates = hybrid_search(
            question=item["question"],
            model=embedding_model,
            collection=collection,
            bm25=bm25,
            documents=indexed_documents,
            top_k=5,
        )

        retrieved_documents = rerank_documents(
            query=item["question"],
            documents=candidates,
            model=reranker,
            top_k=3,
        )

        retrieved_sources = {
            source
            for document in retrieved_documents
            if (
                source := get_document_source(
                    document
                )
            )
        }

        matched_sources = (
            expected_sources & retrieved_sources
        )

        if matched_sources:
            hit_count += 1

        total_expected_sources += len(
            expected_sources
        )

        total_matched_sources += len(
            matched_sources
        )

    total_questions = len(questions)

    hit_rate = (
        hit_count / total_questions
        if total_questions
        else 0.0
    )

    source_recall = (
        total_matched_sources
        / total_expected_sources
        if total_expected_sources
        else 0.0
    )

    return {
        "name": configuration["name"],
        "chunk_size": configuration["chunk_size"],
        "chunk_overlap": configuration["chunk_overlap"],
        "chunks": len(chunks),
        "hit_rate": hit_rate,
        "source_recall": source_recall,
        "matched_sources": total_matched_sources,
        "expected_sources": total_expected_sources,
    }


def main() -> None:
    """Run chunking configuration comparison."""

    questions = load_questions()
    raw_documents = load_documents(DATA_DIRECTORY)

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Loading reranker...")
    reranker = load_reranker()

    results = []

    for configuration in CHUNK_CONFIGURATIONS:
        print(
            "\nEvaluating "
            f"{configuration['name']} configuration..."
        )

        result = evaluate_configuration(
            configuration=configuration,
            questions=questions,
            raw_documents=raw_documents,
            embedding_model=embedding_model,
            reranker=reranker,
        )

        results.append(result)

    print("\n" + "=" * 70)
    print("Chunking Strategy Comparison")
    print("=" * 70)

    for result in results:
        print(
            f"\n{result['name'].upper()}"
            f" ({result['chunk_size']} chars, "
            f"{result['chunk_overlap']} overlap)"
        )

        print(
            f"Generated Chunks: "
            f"{result['chunks']}"
        )

        print(
            f"Retrieval Hit Rate: "
            f"{result['hit_rate']:.2%}"
        )

        print(
            "Expected Sources Retrieved: "
            f"{result['matched_sources']}/"
            f"{result['expected_sources']}"
        )

        print(
            f"Overall Source Recall: "
            f"{result['source_recall']:.2%}"
        )


if __name__ == "__main__":
    main()