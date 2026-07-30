"""Evaluate hallucination resistance on unanswerable questions."""

import json
from pathlib import Path
from typing import Any

from src.rag_pipeline import RAGPipeline


EVALUATION_FILE = Path(
    "evaluation/evaluation_questions.json"
)

EXPECTED_REFUSAL = (
    "I cannot determine the answer from the provided documents."
)


def load_evaluation_questions() -> list[dict[str, Any]]:
    """Load evaluation questions from the JSON dataset."""
    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def is_correct_refusal(answer: str) -> bool:
    """
    Check whether the generated answer correctly refuses
    an unsupported question.

    Args:
        answer:
            Generated answer returned by the RAG pipeline.

    Returns:
        True when the expected refusal statement is present.
    """
    return EXPECTED_REFUSAL.lower() in answer.lower()


def evaluate_generation() -> None:
    """Evaluate hallucination resistance on unanswerable questions."""
    questions = load_evaluation_questions()

    unanswerable_questions = [
        item
        for item in questions
        if not item["answerable"]
    ]

    pipeline = RAGPipeline()
    pipeline.initialize()

    correct_refusals = 0
    evaluated_questions = 0

    print("\nUnanswerable Question Evaluation")
    print("=" * 70)

    for item in unanswerable_questions:
        question = item["question"]

        print(
            f"\nQuestion {item['id']}: "
            f"{question}"
        )
        print("Expected: UNANSWERABLE")

        try:
            result = pipeline.answer(question)

            answer = result["answer"]
            passed = is_correct_refusal(answer)

            evaluated_questions += 1

            if passed:
                correct_refusals += 1

            print("\nAnswer:")
            print(answer)

            print(
                "\nResult:",
                "PASS" if passed else "FAIL",
            )

            print(
                "Citation Confidence:",
                result["citation_confidence"],
            )

        except Exception as error:
            error_message = str(error)

            print(
                f"\nERROR: "
                f"{type(error).__name__}: {error}"
            )

            if (
                "RESOURCE_EXHAUSTED" in error_message
                or "429" in error_message
                or "503" in error_message
                or "UNAVAILABLE" in error_message
            ):
                print(
                    "\nGemini API is temporarily unavailable. "
                    "Evaluation stopped."
                )
                break

        print("-" * 70)

    refusal_rate = (
        correct_refusals / evaluated_questions
        if evaluated_questions
        else 0.0
    )

    print("\n" + "=" * 70)
    print("Generation Evaluation Summary")
    print("=" * 70)

    print(
        "Unanswerable Questions Evaluated: "
        f"{evaluated_questions}/"
        f"{len(unanswerable_questions)}"
    )

    print(
        "Correct Refusals: "
        f"{correct_refusals}/"
        f"{evaluated_questions}"
    )

    print(
        "Hallucination Resistance: "
        f"{refusal_rate:.2%}"
    )


if __name__ == "__main__":
    evaluate_generation()