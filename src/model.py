"""Model training and persistence."""

import logging
import pickle
from pathlib import Path

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork

from .config import MODEL_FILE
from .data import build_network
from .security import verify_file_integrity

logger = logging.getLogger(__name__)


def train_model() -> BayesianNetwork:
    """Build and validate the Bayesian Network model.

    Since this model uses expert-defined CPDs rather than learned parameters,
    'training' consists of building the network structure and validating it.

    Returns:
        Validated BayesianNetwork model
    """
    model = build_network()
    logger.info("Model built successfully.")
    logger.info("Nodes: %s", model.nodes())
    logger.info("Edges: %s", model.edges())
    return model


def save_model(model: BayesianNetwork, path: Path = MODEL_FILE) -> None:
    """Save model to disk.

    Args:
        model: BayesianNetwork to save
        path: File path for saving
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", path)


def load_model(path: Path = MODEL_FILE) -> BayesianNetwork:
    """Load model from disk.

    Args:
        path: File path to load from

    Returns:
        Loaded BayesianNetwork model
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # Verify model integrity before loading
    checksums_file = path.parent / "checksums.json"
    if not verify_file_integrity(path, checksums_file=checksums_file):
        raise ValueError(f"Model file {path} failed integrity check")

    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def get_or_create_model() -> BayesianNetwork:
    """Get existing model or create new one.

    Returns:
        BayesianNetwork model
    """
    try:
        return load_model()
    except FileNotFoundError:
        model = train_model()
        save_model(model)
        return model
