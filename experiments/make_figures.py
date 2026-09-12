"""Render figures/*.png from results/*.json (and the valuation module directly)."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")  # set before pyplot is imported, so this runs headless
import matplotlib.pyplot as plt
from common import AXIS, DEEMPH, FIGURES, INK_2, LABELS, MUTED, RESULTS, SERIES, SURFACE, read_json, style

from figgie.valuation import marginal_value_curve

DPI = 200


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / name)


def fig_marginal_value() -> None:
    fig, (ax_mv, ax_share) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    for color, n in zip(SERIES, (8, 10)):
        curve = marginal_value_curve(n)
        k, mv = curve["k"], curve["marginal_value"]
        label = f"{n}-card goal suit (${200 - 10 * n} bonus)"
        ax_mv.plot(k, mv, color=color, marker="o", ms=6, mec=SURFACE, mew=1.5, label=label, zorder=3)
        peak = int(mv.argmax())
        ax_mv.annotate(
            f"${mv[peak]:.0f} for card {peak + 1}",
            (k[peak], mv[peak]),
            xytext=(9, 2),
            textcoords="offset points",
            color=INK_2,
            fontsize=9,
        )
        ax_share.plot(k, curve["share"], color=color, marker="o", ms=6, mec=SURFACE, mew=1.5, zorder=3)
    ax_mv.axhline(10, color=MUTED, lw=1, zorder=2)
    ax_mv.text(9.2, 5.5, "flat $10 card payout", color=MUTED, fontsize=9, ha="right")
    ax_mv.set(xticks=range(10), ylim=(0, 95), xlabel="Goal-suit cards you already hold",
              ylabel="Value of one more goal card ($)")
    ax_mv.set_title("A goal card's value depends on your inventory")
    ax_mv.legend(loc="upper right")
    ax_share.set(xticks=range(10), ylim=(-0.02, 1.02), xlabel="Goal-suit cards you already hold",
                 ylabel="Expected share of the bonus")
    ax_share.set_title("Chance of the majority bonus")
    fig.text(0.0, -0.04, "Opponents' holdings as dealt (hypergeometric); ties split. "
             "Marginal value = $10 + bonus × change in expected share.", color=MUTED, fontsize=8.5)
    fig.tight_layout()
    save(fig, "marginal_value.png")


def _bars(ax, names, rows, key, highlight, value_fmt="{:+.1f}") -> None:
    y = np.arange(len(names))
    mean = np.array([rows[n][key]["mean"] for n in names])
    lo = np.array([rows[n][key]["lo"] for n in names])
    hi = np.array([rows[n][key]["hi"] for n in names])
    colors = [SERIES[0] if n in highlight else DEEMPH for n in names]
    ax.barh(y, mean, height=0.56, color=colors, zorder=2)
    ax.errorbar(mean, y, xerr=[mean - lo, hi - mean], fmt="none", ecolor=INK_2, elinewidth=1, zorder=3)
    ax.axvline(0, color=AXIS, lw=1, zorder=2)
    span = max(np.abs(lo).max(), np.abs(hi).max())
    for yi, m, l, h in zip(y, mean, lo, hi):
        right = m >= 0
        ax.text(h + span * 0.03 if right else l - span * 0.03, yi, value_fmt.format(m), va="center",
                ha="left" if right else "right", color=INK_2, fontsize=9)
    ax.set_yticks(y, [LABELS[n] for n in names])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(min(lo.min(), 0) - span * 0.22, max(hi.max(), 0) + span * 0.22)


def fig_profit() -> None:
    panels = [(name, read_json(f"{name}.json")) for name in ("headline", "bayes_field") if (RESULTS / f"{name}.json").exists()]
    if not panels:
        return
    first = panels[0][1]["conditions"]
    names = sorted(first, key=lambda n: first[n]["profit"]["mean"])
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 3.9), sharey=True, squeeze=False)
    titles = {"headline": "vs heuristics + hand-only Bayes", "bayes_field": "vs three full Bayesian agents"}
    for ax, (name, data) in zip(axes[0], panels):
        _bars(ax, names, data["conditions"], "profit", highlight={"bayes_flat_value"})
        ax.set_title(titles[name])
        ax.set_xlabel("Chips won per game (mean, 95% CI)")
    n = panels[0][1]["n_seeds"]
    fig.suptitle(f"Profit per game, {n:,} paired deals per agent", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "profit.png")


def fig_convergence() -> None:
    path = RESULTS / "headline.json"
    if not path.exists():
        return
    rows = read_json("headline.json")["conditions"]
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    series = [("bayes", "Exact joint over order flow"), ("bayes_mean_field", "Mean-field over order flow"),
              ("bayes_hand_only", "Hand only")]
    t_max = 0
    for color, (name, label) in zip(SERIES, series):
        curve = rows[name]["belief"]["curve"]
        t, m = np.array(curve["trades"]), np.array(curve["mean"])
        ax.fill_between(t, curve["lo"], curve["hi"], color=color, alpha=0.12, lw=0, zorder=2)
        acc = rows[name]["belief"]["accuracy_end"]["mean"]
        ax.plot(t, m, color=color, label=f"{label}: most-likely suit correct in {acc:.0%} of games", zorder=3)
        t_max = max(t_max, t[-1])
    ax.axhline(0.25, color=MUTED, lw=1, zorder=1)
    ax.text(t_max, 0.215, "no information (1 in 4)", color=MUTED, fontsize=8.5, ha="right", va="top")
    ax.set(ylim=(0, 1), xlim=(0, t_max), xlabel="Trades observed so far",
           ylabel="P(true goal suit), mean over games")
    ax.set_title("Order flow moves beliefs toward the truth; your hand alone can't")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "convergence.png")


def fig_leakage() -> None:
    path = RESULTS / "leakage.json"
    if not path.exists():
        return
    rows = read_json("leakage.json")["conditions"]
    variants = ["conceal_slow", "conceal_decoy", "conceal_both"]
    fig, (ax_profit, ax_belief) = plt.subplots(1, 2, figsize=(10.5, 3.3))
    _bars(ax_profit, variants, rows, "vs_bayes", highlight=set(variants))
    ax_profit.set_title("Profit vs plain Bayesian agent, same deals")
    ax_profit.set_xlabel("Chips per game (paired difference, 95% CI)")
    _bars(ax_belief, variants, rows, "field_p_true_end_vs_bayes", highlight=set(variants), value_fmt="{:+.3f}")
    ax_belief.set_title("What the opponents learned")
    ax_belief.set_xlabel("Change in opponents' final P(true goal)")
    ax_belief.set_yticklabels([])
    n = read_json("leakage.json")["n_seeds"]
    fig.suptitle(f"Concealment against three Bayesian opponents, {n:,} paired deals", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "leakage.png")


def main():
    style()
    fig_marginal_value()
    fig_profit()
    fig_convergence()
    fig_leakage()


if __name__ == "__main__":
    main()
