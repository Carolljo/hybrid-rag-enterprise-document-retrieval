"""FastAPI service for the Hybrid RAG application."""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from api.config import settings
from api.schemas import AnswerResponse, QuestionRequest
from src.rag_pipeline import RAGPipeline


logger = logging.getLogger(__name__)

rag_pipeline = RAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the RAG pipeline once when the API starts.
    """
    logger.info("Initializing RAG pipeline...")

    rag_pipeline.initialize()

    logger.info("RAG API ready.")

    yield


app = FastAPI(
    title=settings.app_name,
    description=(
        "API for answering questions from enterprise documents "
        "using hybrid retrieval, reranking, and grounded generation."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return API and RAG pipeline health status."""
    if rag_pipeline.is_ready():
        return {
            "status": "healthy",
            "rag_pipeline": "ready",
        }

    raise HTTPException(
        status_code=503,
        detail="RAG pipeline is not ready.",
    )


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
        logger.exception(
            "Unexpected error while generating an answer: %s",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer.",
        ) from error