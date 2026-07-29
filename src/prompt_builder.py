"""Prompt construction for grounded RAG answer generation."""

from typing import Any


def format_context(
    documents: list[dict[str, Any]],
) -> str:
    """
    Convert retrieved document chunks into formatted context.

    Args:
        documents:
            Retrieved and reranked document chunks.

    Returns:
        Formatted context containing text and source metadata.
    """
    if not documents:
        return "No relevant context was retrieved."

    context_parts = []

    for index, document in enumerate(documents, start=1):
        source = document["metadata"].get(
            "source",
            "unknown",
        )

        chunk_id = document["metadata"].get(
            "chunk_id",
            "unknown",
        )

        text = document["text"].strip()

        context_parts.append(
            f"[Source {index}]\n"
            f"File: {source}\n"
            f"Chunk: {chunk_id}\n"
            f"Content:\n{text}"
        )

    return "\n\n".join(context_parts)


def build_grounded_prompt(
    question: str,
    documents: list[dict[str, Any]],
) -> str:
    """
    Build a grounded prompt using retrieved documents.

    Args:
        question:
            User question.

        documents:
            Retrieved and reranked document chunks.

    Returns:
        Prompt containing instructions, context,
        and the user question.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    context = format_context(documents)

    prompt = f"""
You are an enterprise document question-answering assistant.

Answer the user's question using ONLY the information contained
in the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent policies, facts, numbers, or procedures.
3. If the provided context does not contain enough information
   to answer the question, clearly say:
   "I cannot determine the answer from the provided documents."
4. Keep the answer concise and directly relevant.
5. When possible, mention the source file that supports the answer.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
""".strip()

    return prompt


if __name__ == "__main__":
    from src.llm import (
        create_client,
        generate_response,
    )

    test_documents = [
        {
            "text": (
                "Remote connections to restricted internal systems "
                "must use the company-approved virtual private "
                "network (VPN). Employees must not disable "
                "endpoint-security software while connected "
                "to company systems."
            ),
            "metadata": {
                "source": "it_security_policy.txt",
                "chunk_id": 1,
            },
        }
    ]

    test_question = "What is the VPN policy?"

    print(f"Question: {test_question}")

    prompt = build_grounded_prompt(
        question=test_question,
        documents=test_documents,
    )

    print("\nCreating Gemini client...")
    client = create_client()

    print("Generating grounded answer...")

    answer = generate_response(
        client=client,
        prompt=prompt,
    )

    print("\nGrounded Answer:")
    print(answer)