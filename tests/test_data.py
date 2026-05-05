"""Tests for data loading and preprocessing."""


from src.config import CPDS
from src.data import build_network, create_cpd, get_variable_info


class TestCreateCPD:
    """Tests for CPD creation."""

    def test_create_marginal_cpd(self):
        """Test creating a marginal (prior) CPD."""
        cpd = create_cpd("F", 2, CPDS["F"])

        assert cpd.variable == "F"
        assert cpd.variable_card == 2
        assert cpd.state_names == {"F": ["V", "F"]}

    def test_create_conditional_cpd(self):
        """Test creating a conditional CPD."""
        cpd = create_cpd("P", 2, CPDS["P"])

        assert cpd.variable == "P"
        assert cpd.variable_card == 2
        assert "F" in cpd.state_names
        assert "E" in cpd.state_names

    def test_cpd_probabilities_sum_to_one(self):
        """Test that CPD columns sum to 1."""
        cpd = create_cpd("A", 2, CPDS["A"])
        values = cpd.get_values()

        # Each column should sum to 1
        for col in range(values.shape[1]):
            assert abs(sum(values[:, col]) - 1.0) < 0.0001


class TestBuildNetwork:
    """Tests for network building."""

    def test_build_network_returns_valid_model(self):
        """Test that build_network creates a valid model."""
        model = build_network()

        assert model.check_model()

    def test_network_has_correct_nodes(self):
        """Test that network has all expected nodes."""
        model = build_network()

        expected_nodes = {"F", "E", "P", "C", "A"}
        assert set(model.nodes()) == expected_nodes

    def test_network_has_correct_edges(self):
        """Test that network has all expected edges."""
        model = build_network()

        expected_edges = {("F", "P"), ("E", "P"), ("E", "C"), ("P", "A")}
        assert set(model.edges()) == expected_edges

    def test_network_is_acyclic(self):
        """Test that the network is a DAG (no cycles)."""
        model = build_network()

        # If model.check_model() passes, it's acyclic
        assert model.check_model()


class TestVariableInfo:
    """Tests for variable information."""

    def test_get_variable_info_returns_dict(self):
        """Test that variable info is returned as dict."""
        info = get_variable_info()

        assert isinstance(info, dict)
        assert len(info) == 5

    def test_variable_info_has_required_fields(self):
        """Test that each variable has required fields."""
        info = get_variable_info()

        for _var_code, var_info in info.items():
            assert "name" in var_info
            assert "states" in var_info
            assert "labels" in var_info
