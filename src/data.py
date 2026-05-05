"""Data loading and preprocessing for the Bayesian Network model."""

from typing import Any

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork

from .config import CPDS, NETWORK_EDGES


def create_cpd(variable: str, cardinality: int, cpd_config: dict[str, Any]) -> TabularCPD:
    """Create a TabularCPD from configuration.

    Args:
        variable: Variable name
        cardinality: Number of states for this variable
        cpd_config: Configuration dictionary with values and state_names

    Returns:
        TabularCPD object
    """
    kwargs = {
        "variable": variable,
        "variable_card": cardinality,
        "values": cpd_config["values"],
        "state_names": cpd_config["state_names"],
    }

    if "evidence" in cpd_config:
        kwargs["evidence"] = cpd_config["evidence"]
        kwargs["evidence_card"] = cpd_config["evidence_card"]

    return TabularCPD(**kwargs)


def build_network() -> BayesianNetwork:
    """Build the Bayesian Network with all CPDs.

    Returns:
        Configured BayesianNetwork object
    """
    model = BayesianNetwork(NETWORK_EDGES)

    cpds = [
        create_cpd("F", 2, CPDS["F"]),
        create_cpd("E", 2, CPDS["E"]),
        create_cpd("P", 2, CPDS["P"]),
        create_cpd("C", 2, CPDS["C"]),
        create_cpd("A", 2, CPDS["A"]),
    ]

    model.add_cpds(*cpds)

    if not model.check_model():
        raise ValueError("Model validation failed. Check CPD definitions.")

    return model


def get_variable_info() -> dict[str, dict[str, Any]]:
    """Get information about model variables.

    Returns:
        Dictionary with variable metadata
    """
    from .config import VARIABLES
    return VARIABLES
