"""Shared agent line-up, result I/O and chart styling for the experiment scripts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figgie.agents import (
    AgentSpec,
    BayesianTrader,
    ConcealingBayesianTrader,
    LongSuitHeuristic,
    RandomAgent,
    ScarcityHeuristic,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Evaluation seeds are disjoint from calibration (10_000+) and tuning (50_000+) seeds.
EVAL_SEED_START = 0


def tuned_kwargs() -> dict[str, dict]:
    """Best execution parameters per agent family from tune_heuristics.py."""
    path = RESULTS / "heuristic_tuning.json"
    if not path.exists():
        raise SystemExit("run experiments/tune_heuristics.py first")
    return {family: row["kwargs"] for family, row in json.loads(path.read_text())["best"].items()}


def agents() -> dict[str, AgentSpec]:
    tuned = tuned_kwargs()
    h_scarcity, h_long, b = tuned["scarcity"], tuned["long_suit"], tuned["bayes"]
    return {
        "random": AgentSpec(RandomAgent, label="random"),
        "scarcity": AgentSpec(ScarcityHeuristic, h_scarcity, label="scarcity"),
        "long_suit": AgentSpec(LongSuitHeuristic, h_long, label="long_suit"),
        "bayes_hand_only": AgentSpec(BayesianTrader, {**b, "use_order_flow": False}, label="bayes_hand_only"),
        "bayes_flat_value": AgentSpec(BayesianTrader, {**b, "inventory_aware": False}, label="bayes_flat_value"),
        "bayes_mean_field": AgentSpec(BayesianTrader, {**b, "exact_joint": False}, label="bayes_mean_field"),
        "bayes": AgentSpec(BayesianTrader, b, label="bayes"),
        "conceal_slow": AgentSpec(ConcealingBayesianTrader, {**b, "cooldown": 5}, label="conceal_slow"),
        "conceal_decoy": AgentSpec(ConcealingBayesianTrader, {**b, "decoy_rate": 0.08}, label="conceal_decoy"),
        "conceal_both": AgentSpec(
            ConcealingBayesianTrader, {**b, "cooldown": 5, "decoy_rate": 0.08}, label="conceal_both"
        ),
    }


def write_json(name: str, payload) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


def read_json(name: str):
    return json.loads((RESULTS / name).read_text())


# ---------------------------------------------------------------------------
# Chart styling (reference palette from the dataviz method, light mode)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua: validated all-pairs
DEEMPH = "#b7d3f6"  # blue step 150, for bars that aren't the story

LABELS = {
    "random": "Random",
    "scarcity": "Scarcity heuristic",
    "long_suit": "Long-suit heuristic",
    "bayes_hand_only": "Bayes, hand only",
    "bayes_flat_value": "Bayes, flat card value",
    "bayes_mean_field": "Bayes, mean-field flow",
    "bayes": "Bayes, full",
    "conceal_slow": "Slowed buying",
    "conceal_decoy": "Decoy bids",
    "conceal_both": "Slowed + decoys",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )
