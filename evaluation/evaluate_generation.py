"""Evaluate hallucination resistance on unanswerable questions."""

import json
import time
from pathlib import Path
from typing import Any

from src.rag_pipeline import RAGPipeline


EVALUATION_FILE = Path(
    "evaluation/evaluation_questions.json"
)

REQUEST_DELAY_SECONDS = 20


def load_evaluation_questions() -> list[dict[str, Any]]:
    """Load evaluation questions."""
    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def evaluate_generation() -> None:
    """Evaluate generation on unanswerable questions."""
    questions = load_evaluation_questions()

    unanswerable_questions = [
        item
        for item in questions
        if not item["answerable"]
    ]

    pipeline = RAGPipeline()
    pipeline.initialize()

    print("\nUnanswerable Question Evaluation")
    print("=" * 70)

    for index, item in enumerate(
        unanswerable_questions,
        start=1,
    ):
        question = item["question"]

        print(f"\nQuestion {item['id']}: {question}")
        print("Expected: UNANSWERABLE")

        try:
            result = pipeline.answer(question)

            print("\nAnswer:")
            print(result["answer"])

            print("\nCitations:")

            citations = result["citations"]

            if citations:
                for citation in citations:
                    print(
                        f"- {citation['source']} "
                        f"(Chunk {citation['chunk_id']})"
                    )
            else:
                print("- None")

        except Exception as error:
            error_message = str(error)

            print(
                f"\nERROR: "
                f"{type(error).__name__}: {error}"
            )

            if (
                "RESOURCE_EXHAUSTED" in error_message
                or "429" in error_message
            ):
                print("\nGemini API quota exhausted.")
                print(
                    "Evaluation stopped to avoid "
                    "unnecessary API requests."
                )
                print("-" * 70)
                break

        print("-" * 70)

        if index < len(unanswerable_questions):
            print(
                f"Waiting {REQUEST_DELAY_SECONDS} seconds "
                "before the next request..."
            )
            time.sleep(REQUEST_DELAY_SECONDS)


if __name__ == "__main__":
    evaluate_generation()