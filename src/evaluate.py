"""Evaluation and inference for the Bayesian Network model."""

from typing import Any

from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork

from .config import VARIABLES


def create_inference_engine(model: BayesianNetwork) -> VariableElimination:
    """Create an inference engine for the model.

    Args:
        model: BayesianNetwork model

    Returns:
        VariableElimination inference object
    """
    return VariableElimination(model)


def query(
    model: BayesianNetwork,
    target: str,
    evidence: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Perform a probabilistic query on the model.

    Args:
        model: BayesianNetwork model
        target: Variable to query (e.g., "A" for Heart Attack)
        evidence: Dictionary of observed variables (e.g., {"F": "V"} for Smoker=Yes)

    Returns:
        Dictionary with state probabilities
    """
    infer = create_inference_engine(model)
    result = infer.query([target], evidence=evidence or {})

    states = result.state_names[target]
    probabilities = result.values.tolist()

    return {
        "variable": target,
        "variable_name": VARIABLES[target]["name"],
        "evidence": evidence or {},
        "probabilities": {state: round(prob, 4) for state, prob in zip(states, probabilities, strict=False)},
    }


def predict_heart_attack(
    model: BayesianNetwork,
    smoker: bool | None = None,
    exercise: bool | None = None,
    high_cholesterol: bool | None = None,
    high_pressure: bool | None = None,
) -> dict[str, Any]:
    """Predict heart attack probability given risk factors.

    Args:
        model: BayesianNetwork model
        smoker: True if patient smokes
        exercise: True if patient exercises regularly
        high_cholesterol: True if patient has high cholesterol
        high_pressure: True if patient has high blood pressure

    Returns:
        Prediction result with probabilities
    """
    evidence = {}

    if smoker is not None:
        evidence["F"] = "V" if smoker else "F"
    if exercise is not None:
        evidence["E"] = "V" if exercise else "F"
    if high_cholesterol is not None:
        evidence["C"] = "A" if high_cholesterol else "B"
    if high_pressure is not None:
        evidence["P"] = "A" if high_pressure else "B"

    result = query(model, "A", evidence)

    prob_attack = result["probabilities"]["A"]
    prob_no_attack = result["probabilities"]["B"]

    return {
        "prediction": "Heart Attack Likely" if prob_attack > 0.5 else "Heart Attack Unlikely",
        "probability_attack": prob_attack,
        "probability_no_attack": prob_no_attack,
        "risk_level": _get_risk_level(prob_attack),
        "evidence_provided": {
            "smoker": smoker,
            "exercise": exercise,
            "high_cholesterol": high_cholesterol,
            "high_pressure": high_pressure,
        },
    }


def _get_risk_level(probability: float) -> str:
    """Determine risk level from probability."""
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"


def run_validation_queries(model: BayesianNetwork) -> list[dict[str, Any]]:
    """Run the three validation queries from the notebook.

    Args:
        model: BayesianNetwork model

    Returns:
        List of query results
    """
    queries = [
        {"name": "P(A|F=V)", "target": "A", "evidence": {"F": "V"}, "expected": 0.575, "state": "A"},
        {"name": "P(P|A=A)", "target": "P", "evidence": {"A": "A"}, "expected": 0.9125, "state": "A"},
        {"name": "P(A|F=V,C=A)", "target": "A", "evidence": {"F": "V", "C": "A"}, "expected": 0.6275, "state": "A"},
    ]

    results = []
    for q in queries:
        result = query(model, q["target"], q["evidence"])
        actual = result["probabilities"][q["state"]]
        passed = abs(actual - q["expected"]) < 0.0001

        results.append({
            "query": q["name"],
            "expected": q["expected"],
            "actual": actual,
            "passed": passed,
        })

    return results
