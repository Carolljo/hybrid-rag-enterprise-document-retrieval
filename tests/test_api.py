
"""Tests for the Hybrid RAG FastAPI endpoints."""

from collections.abc import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app, rag_pipeline


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client without initializing the real RAG pipeline."""
    with patch.object(
        rag_pipeline,
        "initialize",
        return_value=None,
    ):
        with TestClient(app) as test_client:
            yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """Health endpoint should report a ready pipeline."""
    with patch.object(
        rag_pipeline,
        "is_ready",
        return_value=True,
    ):
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "rag_pipeline": "ready",
    }


def test_health_endpoint_not_ready(client: TestClient) -> None:
    """Health endpoint should return 503 when pipeline is not ready."""
    with patch.object(
        rag_pipeline,
        "is_ready",
        return_value=False,
    ):
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "RAG pipeline is not ready.",
    }


def test_ask_endpoint(client: TestClient) -> None:
    """Ask endpoint should return an answer with verified citations."""
    mock_result = {
        "question": "What is the VPN policy?",
        "answer": "Employees must use the approved corporate VPN.",
        "citations": [
            {
                "source": "it_security_policy.txt",
                "chunk_id": 1,
                "verified": True,
            }
        ],
        "citation_confidence": 1.0,
    }

    with patch.object(
        rag_pipeline,
        "answer",
        return_value=mock_result,
    ):
        response = client.post(
            "/ask",
            json={
                "question": "What is the VPN policy?",
            },
        )

    assert response.status_code == 200
    assert response.json() == mock_result

def test_ask_endpoint_empty_question(client: TestClient) -> None:
    """Ask endpoint should reject an empty question."""
    response = client.post(
        "/ask",
        json={
            "question": "",
        },
    )

    assert response.status_code == 422


def test_ask_endpoint_missing_question(client: TestClient) -> None:
    """Ask endpoint should reject a request without a question."""
    response = client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422


def test_ask_endpoint_invalid_question_type(
    client: TestClient,
) -> None:
    """Ask endpoint should reject an invalid question type."""
    response = client.post(
        "/ask",
        json={
            "question": ["invalid", "question"],
        },
    )

    assert response.status_code == 422


def test_ask_endpoint_pipeline_failure(
    client: TestClient,
) -> None:
    """Ask endpoint should return 500 for unexpected pipeline errors."""
    with patch.object(
        rag_pipeline,
        "answer",
        side_effect=RuntimeError("Pipeline failure."),
    ):
        response = client.post(
            "/ask",
            json={
                "question": "What is the VPN policy?",
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Unable to generate an answer.",
    }
