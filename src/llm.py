"""Gemini LLM integration for grounded answer generation."""

from google import genai

from api.config import settings


DEFAULT_MODEL = "gemini-3.6-flash"


def create_client() -> genai.Client:
    """
    Create a Gemini client using the API key
    from the centralized application settings.

    Returns:
        Configured Gemini client.
    """
    return genai.Client(
        api_key=settings.gemini_api_key
    )


def generate_response(
    client: genai.Client,
    prompt: str,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """
    Generate a response using Gemini.

    Args:
        client:
            Configured Gemini client.

        prompt:
            Prompt sent to the language model.

        model_name:
            Gemini model used for generation.

    Returns:
        Generated text response.
    """
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response.text.strip()


if __name__ == "__main__":
    print("Creating Gemini client...")

    client = create_client()

    test_prompt = (
        "Reply with exactly this sentence: "
        "Gemini connection successful."
    )

    print("Sending test request...")

    answer = generate_response(
        client=client,
        prompt=test_prompt,
    )

    print("\nGemini Response:")
    print(answer)