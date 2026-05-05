"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from api.app import app
from src.model import train_model


@pytest.fixture(scope="module")
def client():
    """Create a test client with model loaded."""
    # Load model for testing
    import api.app as app_module
    app_module.model = train_model()

    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")

        assert response.status_code == 200

    def test_health_response_format(self, client):
        """Test health response has expected fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_smoker(self, client):
        """Test prediction for smoker."""
        response = client.post("/predict", json={"smoker": True})

        assert response.status_code == 200
        data = response.json()
        assert abs(data["probability_attack"] - 0.575) < 0.01
        assert data["prediction"] == "Heart Attack Likely"

    def test_predict_with_multiple_factors(self, client):
        """Test prediction with multiple risk factors."""
        response = client.post("/predict", json={
            "smoker": True,
            "high_cholesterol": True,
        })

        assert response.status_code == 200
        data = response.json()
        assert abs(data["probability_attack"] - 0.6275) < 0.01

    def test_predict_empty_evidence(self, client):
        """Test prediction with no evidence."""
        response = client.post("/predict", json={})

        assert response.status_code == 200
        data = response.json()
        assert "probability_attack" in data

    def test_predict_response_structure(self, client):
        """Test prediction response has all expected fields."""
        response = client.post("/predict", json={"smoker": True})
        data = response.json()

        assert "prediction" in data
        assert "probability_attack" in data
        assert "probability_no_attack" in data
        assert "risk_level" in data
        assert "evidence_provided" in data


class TestQueryEndpoint:
    """Tests for custom query endpoint."""

    def test_query_heart_attack(self, client):
        """Test custom query for heart attack."""
        response = client.post("/query", json={
            "target": "A",
            "evidence": {"F": "V"},
        })

        assert response.status_code == 200
        data = response.json()
        assert data["variable"] == "A"
        assert abs(data["probabilities"]["A"] - 0.575) < 0.01

    def test_query_pressure_given_attack(self, client):
        """Test query for P(P|A=A)."""
        response = client.post("/query", json={
            "target": "P",
            "evidence": {"A": "A"},
        })

        assert response.status_code == 200
        data = response.json()
        assert abs(data["probabilities"]["A"] - 0.9125) < 0.01

    def test_query_invalid_target(self, client):
        """Test query with invalid target variable."""
        response = client.post("/query", json={
            "target": "X",
            "evidence": {},
        })

        assert response.status_code == 400

    def test_query_invalid_evidence(self, client):
        """Test query with invalid evidence variable."""
        response = client.post("/query", json={
            "target": "A",
            "evidence": {"X": "V"},
        })

        assert response.status_code == 400


class TestVariablesEndpoint:
    """Tests for variables info endpoint."""

    def test_variables_returns_200(self, client):
        """Test variables endpoint returns 200."""
        response = client.get("/variables")

        assert response.status_code == 200

    def test_variables_has_all_vars(self, client):
        """Test all variables are returned."""
        response = client.get("/variables")
        data = response.json()

        expected = {"A", "P", "C", "F", "E"}
        assert set(data.keys()) == expected
