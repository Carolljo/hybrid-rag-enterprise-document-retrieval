"""FastAPI service for the Hybrid RAG application."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag_pipeline import RAGPipeline


class QuestionRequest(BaseModel):
    """Request body for document question answering."""

    question: str = Field(
        ...,
        min_length=1,
        description="Question to answer using enterprise documents.",
    )


class CitationResponse(BaseModel):
    """Citation returned with a generated answer."""

    source: str
    chunk_id: int | str


class AnswerResponse(BaseModel):
    """Response returned by the RAG API."""

    question: str
    answer: str
    citations: list[CitationResponse]


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