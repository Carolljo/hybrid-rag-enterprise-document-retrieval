"""Pydantic schemas for the Hybrid RAG API."""

from pydantic import BaseModel, Field


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
    verified: bool


class AnswerResponse(BaseModel):
    """Response returned by the RAG API."""

    question: str
    answer: str
    citations: list[CitationResponse]
    citation_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Proportion of returned citations verified against "
            "retrieved document chunks."
        ),
    )