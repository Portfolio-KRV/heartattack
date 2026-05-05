"""PNG: heart attack Bayesian Network DAG."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

LABELS = {
    "F": "Smoker",
    "E": "Exercise",
    "P": "High Pressure",
    "C": "High Cholesterol",
    "A": "Heart Attack",
}
EDGES = [("F", "P"), ("E", "P"), ("E", "C"), ("P", "A")]


def main(out: Path) -> int:
    G = nx.DiGraph()
    G.add_edges_from(EDGES)

    pos = {"F": (-2, 2), "E": (1, 2), "P": (-1, 0.5), "C": (2, 0.5), "A": (-1, -1.2)}

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=110)

    node_colors = ["#dc2626" if n == "A" else "#6366f1" for n in G.nodes()]
    sizes = [3500 if n == "A" else 2400 for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, edge_color="#94a3b8", arrows=True,
                           arrowsize=22, width=1.6, node_size=2400,
                           connectionstyle="arc3,rad=0.05", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes,
                           edgecolors="white", linewidths=2, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: LABELS[n] for n in G.nodes()},
                            font_size=10, font_color="white", font_weight="bold", ax=ax)

    ax.set_title("Bayesian Network — heart attack risk DAG (expert-defined CPDs)",
                 fontsize=13, weight="bold")
    ax.text(0.5, -0.05,
            "Each arrow = a conditional probability. Given evidence on any node, "
            "P(Heart Attack | evidence) is computed via belief propagation.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color="#475569")
    ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).parents[2] / "portfolio-website" / "public" / "previews" / "heartattack.png"
    )
    sys.exit(main(out))
