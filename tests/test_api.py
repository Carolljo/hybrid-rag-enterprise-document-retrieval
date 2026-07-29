"""Tests for the Hybrid RAG FastAPI endpoints."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app, rag_pipeline


def test_health_endpoint() -> None:
    """Health endpoint should report a ready pipeline."""
    with patch.object(
        rag_pipeline,
        "is_ready",
        return_value=True,
    ):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "rag_pipeline": "ready",
    }


def test_ask_endpoint() -> None:
    """Ask endpoint should return an answer and citations."""
    mock_result = {
        "question": "What is the VPN policy?",
        "answer": "Employees must use the approved corporate VPN.",
        "citations": [
            {
                "source": "it_security_policy.txt",
                "chunk_id": 1,
            }
        ],
    }

    with patch.object(
        rag_pipeline,
        "answer",
        return_value=mock_result,
    ):
        with TestClient(app) as client:
            response = client.post(
                "/ask",
                json={
                    "question": "What is the VPN policy?",
                },
            )

    assert response.status_code == 200
    assert response.json() == mock_result