"""Evaluate retrieval quality of the Hybrid RAG pipeline."""

import json
from pathlib import Path
from typing import Any

from src.rag_pipeline import RAGPipeline


EVALUATION_FILE = Path("evaluation/evaluation_questions.json")


def load_evaluation_questions() -> list[dict[str, Any]]:
    """Load evaluation questions from the JSON dataset."""
    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_document_source(
    document: dict[str, Any],
) -> str | None:
    """Extract the source filename from a retrieved document."""
    source = document.get("source")

    if source:
        return str(source)

    metadata = document.get("metadata", {})

    if isinstance(metadata, dict):
        source = metadata.get("source")

        if source:
            return str(source)

    return None


def evaluate_retrieval() -> None:
    """Evaluate whether retrieval returns expected source documents."""
    questions = load_evaluation_questions()

    pipeline = RAGPipeline()
    pipeline.initialize()

    answerable_questions = [
        item
        for item in questions
        if item["answerable"]
    ]

    correct = 0

    print("\nRetrieval Evaluation")
    print("=" * 70)

    for item in answerable_questions:
        question = item["question"]
        expected_sources = set(
            item["expected_sources"]
        )

        retrieved_documents = pipeline.retrieve(
            question
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

        passed = bool(matched_sources)

        if passed:
            correct += 1

        print(
            f"\nQuestion {item['id']}: "
            f"{question}"
        )

        print(
            "Expected sources:",
            sorted(expected_sources),
        )

        print(
            "Retrieved sources:",
            sorted(retrieved_sources),
        )

        print(
            "Matched sources:",
            sorted(matched_sources),
        )

        print(
            "Result:",
            "PASS" if passed else "FAIL",
        )

    total = len(answerable_questions)

    hit_rate = (
        correct / total
        if total
        else 0.0
    )

    print("\n" + "=" * 70)
    print("Retrieval Evaluation Summary")
    print("=" * 70)

    print(
        f"Correct: {correct}/{total}"
    )

    print(
        f"Retrieval Hit Rate: "
        f"{hit_rate:.2%}"
    )


if __name__ == "__main__":
    evaluate_retrieval()

