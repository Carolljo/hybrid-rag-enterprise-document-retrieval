"""FastAPI service for the Hybrid RAG application."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from api.schemas import AnswerResponse, QuestionRequest
from src.rag_pipeline import RAGPipeline


rag_pipeline = RAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the RAG pipeline once when the API starts.
    """
    print("Initializing RAG pipeline...")

    rag_pipeline.initialize()

    print("RAG API ready.")

    yield


app = FastAPI(
    title="Hybrid RAG Enterprise Document Retrieval API",
    description=(
        "API for answering questions from enterprise documents "
        "using hybrid retrieval, reranking, and grounded generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return API health status."""
    return {
        "status": "healthy",
    }


@app.post(
    "/ask",
    response_model=AnswerResponse,
)
def ask_question(
    request: QuestionRequest,
) -> dict[str, Any]:
    """
    Answer a question using the Hybrid RAG pipeline.
    """
    try:
        result = rag_pipeline.answer(
            request.question
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer.",
        ) from error