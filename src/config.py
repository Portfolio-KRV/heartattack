"""Configuration for the Heart Attack prediction model."""

from pathlib import Path
from typing import Any

# Project paths
PROJECT_ROOT: Path = Path(__file__).parent.parent
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Model file
MODEL_FILE: Path = MODELS_DIR / "bayesian_network.pkl"

# Variable names and their states
VARIABLES: dict[str, dict[str, Any]] = {
    "F": {"name": "Smoker", "states": ["V", "F"], "labels": {"V": "Yes", "F": "No"}},
    "E": {"name": "Exercise", "states": ["V", "F"], "labels": {"V": "Yes", "F": "No"}},
    "P": {"name": "High Pressure", "states": ["A", "B"], "labels": {"A": "Yes", "B": "No"}},
    "C": {"name": "High Cholesterol", "states": ["A", "B"], "labels": {"A": "Yes", "B": "No"}},
    "A": {"name": "Heart Attack", "states": ["A", "B"], "labels": {"A": "Yes", "B": "No"}},
}

# Network structure (edges in the DAG)
NETWORK_EDGES: list[tuple[str, str]] = [
    ("F", "P"),  # Smoker -> High Pressure
    ("E", "P"),  # Exercise -> High Pressure
    ("E", "C"),  # Exercise -> High Cholesterol
    ("P", "A"),  # High Pressure -> Heart Attack
]

# Conditional Probability Tables (expert-defined)
CPDS: dict[str, dict[str, Any]] = {
    "F": {
        "values": [[0.15], [0.85]],
        "state_names": {"F": ["V", "F"]},
    },
    "E": {
        "values": [[0.4], [0.6]],
        "state_names": {"E": ["V", "F"]},
    },
    "P": {
        "values": [[0.45, 0.95, 0.05, 0.55], [0.55, 0.05, 0.95, 0.45]],
        "evidence": ["F", "E"],
        "evidence_card": [2, 2],
        "state_names": {"P": ["A", "B"], "F": ["V", "F"], "E": ["V", "F"]},
    },
    "C": {
        "values": [[0.4, 0.8], [0.6, 0.2]],
        "evidence": ["E"],
        "evidence_card": [2],
        "state_names": {"C": ["A", "B"], "E": ["V", "F"]},
    },
    "A": {
        "values": [[0.75, 0.05], [0.25, 0.95]],
        "evidence": ["P"],
        "evidence_card": [2],
        "state_names": {"A": ["A", "B"], "P": ["A", "B"]},
    },
}
