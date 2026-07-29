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
    """Evaluate retrieval against expected source documents."""
    questions = load_evaluation_questions()

    pipeline = RAGPipeline()
    pipeline.initialize()

    answerable_questions = [
        item
        for item in questions
        if item["answerable"]
    ]

    hit_count = 0
    total_expected_sources = 0
    total_matched_sources = 0

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
            hit_count += 1

        total_expected_sources += len(
            expected_sources
        )

        total_matched_sources += len(
            matched_sources
        )

        question_recall = (
            len(matched_sources)
            / len(expected_sources)
            if expected_sources
            else 0.0
        )

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
            f"Source Recall: "
            f"{question_recall:.2%}"
        )

        print(
            "Result:",
            "PASS" if passed else "FAIL",
        )

    total_questions = len(
        answerable_questions
    )

    hit_rate = (
        hit_count / total_questions
        if total_questions
        else 0.0
    )

    overall_source_recall = (
        total_matched_sources
        / total_expected_sources
        if total_expected_sources
        else 0.0
    )

    print("\n" + "=" * 70)
    print("Retrieval Evaluation Summary")
    print("=" * 70)

    print(
        f"Questions with relevant source: "
        f"{hit_count}/{total_questions}"
    )

    print(
        f"Retrieval Hit Rate: "
        f"{hit_rate:.2%}"
    )

    print(
        f"Expected sources retrieved: "
        f"{total_matched_sources}/"
        f"{total_expected_sources}"
    )

    print(
        f"Overall Source Recall: "
        f"{overall_source_recall:.2%}"
    )


if __name__ == "__main__":
    evaluate_retrieval()