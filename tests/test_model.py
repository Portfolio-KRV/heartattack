"""Tests for model training and inference."""

import tempfile
from pathlib import Path

import pytest

from src.evaluate import predict_heart_attack, query, run_validation_queries
from src.model import load_model, save_model, train_model


class TestTrainModel:
    """Tests for model training."""

    def test_train_model_returns_model(self):
        """Test that train_model returns a valid model."""
        model = train_model()

        assert model is not None
        assert model.check_model()

    def test_train_model_is_reproducible(self):
        """Test that training produces consistent models."""
        model1 = train_model()
        model2 = train_model()

        assert set(model1.nodes()) == set(model2.nodes())
        assert set(model1.edges()) == set(model2.edges())


class TestModelPersistence:
    """Tests for model saving and loading."""

    def test_save_and_load_model(self):
        """Test that model can be saved and loaded."""
        model = train_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.pkl"
            save_model(model, path)
            loaded = load_model(path)

            assert loaded.check_model()
            assert set(loaded.nodes()) == set(model.nodes())

    def test_load_nonexistent_model_raises(self):
        """Test that loading nonexistent model raises error."""
        with pytest.raises(FileNotFoundError):
            load_model(Path("/nonexistent/path/model.pkl"))


class TestQuery:
    """Tests for probabilistic queries."""

    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        return train_model()

    def test_query_smoker_heart_attack(self, model):
        """Test P(A|F=V) = 0.575."""
        result = query(model, "A", {"F": "V"})

        assert abs(result["probabilities"]["A"] - 0.575) < 0.0001

    def test_query_pressure_given_attack(self, model):
        """Test P(P|A=A) = 0.9125."""
        result = query(model, "P", {"A": "A"})

        assert abs(result["probabilities"]["A"] - 0.9125) < 0.0001

    def test_query_attack_given_smoker_cholesterol(self, model):
        """Test P(A|F=V,C=A) = 0.6275."""
        result = query(model, "A", {"F": "V", "C": "A"})

        assert abs(result["probabilities"]["A"] - 0.6275) < 0.0001

    def test_query_without_evidence(self, model):
        """Test query without evidence returns marginal."""
        result = query(model, "A")

        assert "A" in result["probabilities"]
        assert "B" in result["probabilities"]
        # Probabilities should sum to 1
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.0001


class TestPredictHeartAttack:
    """Tests for prediction function."""

    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        return train_model()

    def test_predict_smoker(self, model):
        """Test prediction for smoker."""
        result = predict_heart_attack(model, smoker=True)

        assert result["probability_attack"] == pytest.approx(0.575, abs=0.001)
        assert result["prediction"] == "Heart Attack Likely"
        assert result["risk_level"] == "Medium"

    def test_predict_non_smoker_exerciser(self, model):
        """Test prediction for healthy profile."""
        result = predict_heart_attack(model, smoker=False, exercise=True)

        assert result["probability_attack"] < 0.5
        assert result["prediction"] == "Heart Attack Unlikely"

    def test_predict_high_risk(self, model):
        """Test prediction with high blood pressure."""
        result = predict_heart_attack(model, high_pressure=True)

        assert result["probability_attack"] == pytest.approx(0.75, abs=0.001)
        assert result["risk_level"] == "High"

    def test_predict_low_risk(self, model):
        """Test prediction with normal blood pressure."""
        result = predict_heart_attack(model, high_pressure=False)

        assert result["probability_attack"] == pytest.approx(0.05, abs=0.001)
        assert result["risk_level"] == "Low"


class TestValidationQueries:
    """Tests for validation queries from notebook."""

    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        return train_model()

    def test_all_validation_queries_pass(self, model):
        """Test that all validation queries from notebook pass."""
        results = run_validation_queries(model)

        for result in results:
            assert result["passed"], f"Query {result['query']} failed"
