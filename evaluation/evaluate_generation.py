"""Evaluate generation quality of the Hybrid RAG pipeline."""

import argparse
import json
from pathlib import Path
from typing import Any

from sentence_transformers import util

from src.embeddings import load_embedding_model
from src.rag_pipeline import RAGPipeline


EVALUATION_FILE = Path(
    "evaluation/evaluation_questions.json"
)

EXPECTED_REFUSAL = (
    "I cannot determine the answer from the provided documents."
)

SIMILARITY_THRESHOLD = 0.70


def load_evaluation_questions() -> list[dict[str, Any]]:
    """Load evaluation questions from the JSON dataset."""
    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def parse_arguments() -> argparse.Namespace:
    """Parse optional question-range arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid RAG generation quality."
    )

    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="First question ID to evaluate.",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last question ID to evaluate.",
    )

    return parser.parse_args()


def select_questions(
    questions: list[dict[str, Any]],
    start: int,
    end: int | None,
) -> list[dict[str, Any]]:
    """Select evaluation questions by inclusive ID range."""
    if start < 1:
        raise ValueError("--start must be at least 1.")

    if end is not None and end < start:
        raise ValueError("--end must be greater than or equal to --start.")

    return [
        item
        for item in questions
        if (
            item["id"] >= start
            and (
                end is None
                or item["id"] <= end
            )
        )
    ]


def is_correct_refusal(answer: str) -> bool:
    """Check whether an unsupported question was refused."""
    return EXPECTED_REFUSAL.lower() in answer.lower()


def calculate_answer_similarity(
    generated_answer: str,
    expected_answer: str,
    model: Any,
) -> float:
    """
    Calculate semantic similarity between generated
    and golden reference answers.
    """
    embeddings = model.encode(
        [
            generated_answer,
            expected_answer,
        ],
        convert_to_tensor=True,
    )

    similarity = util.cos_sim(
        embeddings[0],
        embeddings[1],
    ).item()

    return round(similarity, 3)


def evaluate_generation(
    questions: list[dict[str, Any]],
) -> None:
    """Evaluate answer correctness and hallucination resistance."""
    if not questions:
        raise ValueError(
            "No evaluation questions matched the requested range."
        )

    pipeline = RAGPipeline()
    pipeline.initialize()

    evaluation_model = load_embedding_model()

    answerable_count = 0
    answerable_passes = 0
    similarity_total = 0.0

    refusal_count = 0
    refusal_passes = 0

    type_results: dict[str, dict[str, float]] = {}

    print("\nGeneration Evaluation")
    print("=" * 70)

    print(
        "Question range: "
        f"{questions[0]['id']}–{questions[-1]['id']}"
    )

    for item in questions:
        question = item["question"]
        question_type = item["type"]

        print(
            f"\nQuestion {item['id']} "
            f"[{question_type}]: "
            f"{question}"
        )

        try:
            result = pipeline.answer(question)
            answer = result["answer"]

            print("\nGenerated Answer:")
            print(answer)

            if item["answerable"]:
                expected_answer = item["expected_answer"]

                similarity = calculate_answer_similarity(
                    generated_answer=answer,
                    expected_answer=expected_answer,
                    model=evaluation_model,
                )

                passed = (
                    similarity >= SIMILARITY_THRESHOLD
                )

                answerable_count += 1
                similarity_total += similarity

                if passed:
                    answerable_passes += 1

                if question_type not in type_results:
                    type_results[question_type] = {
                        "count": 0,
                        "passes": 0,
                        "similarity": 0.0,
                    }

                type_results[question_type]["count"] += 1
                type_results[question_type][
                    "similarity"
                ] += similarity

                if passed:
                    type_results[question_type][
                        "passes"
                    ] += 1

                print("\nExpected Answer:")
                print(expected_answer)

                print(
                    f"\nSemantic Similarity: "
                    f"{similarity:.3f}"
                )

                print(
                    "Result:",
                    "PASS" if passed else "FAIL",
                )

            else:
                passed = is_correct_refusal(answer)

                refusal_count += 1

                if passed:
                    refusal_passes += 1

                print("\nExpected: UNANSWERABLE")

                print(
                    "Result:",
                    "PASS" if passed else "FAIL",
                )

            print(
                "Citation Confidence:",
                f"{result['citation_confidence']:.1%}",
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
                    "\nGemini API is temporarily unavailable "
                    "or quota-limited. Evaluation stopped."
                )
                break

        print("-" * 70)

    average_similarity = (
        similarity_total / answerable_count
        if answerable_count
        else 0.0
    )

    correctness_rate = (
        answerable_passes / answerable_count
        if answerable_count
        else 0.0
    )

    refusal_rate = (
        refusal_passes / refusal_count
        if refusal_count
        else 0.0
    )

    print("\n" + "=" * 70)
    print("Generation Evaluation Summary")
    print("=" * 70)

    print(
        "Answerable Questions Evaluated: "
        f"{answerable_count}"
    )

    if answerable_count:
        print(
            "Answer Correctness Pass Rate: "
            f"{correctness_rate:.2%}"
        )

        print(
            "Average Semantic Similarity: "
            f"{average_similarity:.3f}"
        )
    else:
        print(
            "Answer Correctness Pass Rate: "
            "N/A"
        )

        print(
            "Average Semantic Similarity: "
            "N/A"
        )

    print("\nResults by Question Type")

    if not type_results:
        print("- No answerable questions evaluated.")

    for question_type, values in type_results.items():
        count = int(values["count"])
        passes = int(values["passes"])

        pass_rate = (
            passes / count
            if count
            else 0.0
        )

        average_type_similarity = (
            values["similarity"] / count
            if count
            else 0.0
        )

        print(
            f"- {question_type}: "
            f"{passes}/{count} passed "
            f"({pass_rate:.2%}), "
            f"average similarity "
            f"{average_type_similarity:.3f}"
        )

    print(
        "\nUnanswerable Questions Evaluated: "
        f"{refusal_count}"
    )

    if refusal_count:
        print(
            "Correct Refusals: "
            f"{refusal_passes}/{refusal_count}"
        )

        print(
            "Hallucination Resistance: "
            f"{refusal_rate:.2%}"
        )
    else:
        print("Correct Refusals: N/A")
        print("Hallucination Resistance: N/A")


def main() -> None:
    """Run generation evaluation for the requested range."""
    args = parse_arguments()

    questions = load_evaluation_questions()

    selected_questions = select_questions(
        questions=questions,
        start=args.start,
        end=args.end,
    )

    evaluate_generation(selected_questions)


if __name__ == "__main__":
    main()