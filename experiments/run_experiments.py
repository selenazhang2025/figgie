"""Run the paired-seed experiments and write results/*.json. Plots come from make_figures.py.

  headline     every agent as focal against a mixed field (tuned heuristics + hand-only Bayes)
  bayes_field  the same focal agents against three full Bayesian opponents
  leakage      information-concealing variants against three full Bayesian opponents

Every condition in an experiment plays the same seeds, so each focal agent faces the
identical deal, seat and opponents.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from common import EVAL_SEED_START, agents, write_json

from figgie.tournament import FocalRecord, mean_ci, paired_diff_ci, run_focal

TRADE_HORIZON = 45
MIN_GAMES_FOR_CURVE = 200
MIN_SHARE_FOR_CURVE = 0.25

HEADLINE_CONDITIONS = ["random", "scarcity", "long_suit", "bayes_hand_only", "bayes_flat_value", "bayes_mean_field", "bayes"]
LEAKAGE_CONDITIONS = ["bayes", "conceal_slow", "conceal_decoy", "conceal_both"]


def est(e) -> dict:
    return {"mean": e.mean, "lo": e.lo, "hi": e.hi, "n": e.n}


def belief_path(diag: dict, goal: int, n_trades: int) -> np.ndarray:
    """P(true goal) after observing t trades, t = 0..TRADE_HORIZON; NaN past the game's last trade."""
    trades, probs = diag["belief_trades"], diag["goal_probs"][:, goal]
    t = np.arange(TRADE_HORIZON + 1)
    path = probs[np.searchsorted(trades, t, side="right") - 1].astype(np.float64)
    path[t > n_trades] = np.nan
    return path


def belief_summary(records: list[FocalRecord], diag_of) -> dict:
    paths, start, end, correct = [], [], [], []
    for r in records:
        for diag in diag_of(r):
            if "goal_probs" not in diag:
                continue
            probs = diag["goal_probs"]
            paths.append(belief_path(diag, r.goal_suit, r.n_trades))
            start.append(probs[0, r.goal_suit])
            end.append(probs[-1, r.goal_suit])
            correct.append(probs[-1].argmax() == r.goal_suit)
    if not paths:
        return {}
    paths = np.array(paths)
    n = np.sum(~np.isnan(paths), axis=0)
    mean = np.nanmean(paths, axis=0)
    half = 1.96 * np.nanstd(paths, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    # Later trade counts are only reached in busier games; stop before the tail is a biased few.
    keep = n >= max(MIN_GAMES_FOR_CURVE, MIN_SHARE_FOR_CURVE * len(paths))
    return {
        "p_true_start": est(mean_ci(start)),
        "p_true_end": est(mean_ci(end)),
        "accuracy_end": est(mean_ci(np.array(correct, dtype=float))),
        "curve": {
            "trades": np.arange(TRADE_HORIZON + 1)[keep].tolist(),
            "mean": mean[keep].tolist(),
            "lo": (mean - half)[keep].tolist(),
            "hi": (mean + half)[keep].tolist(),
            "n": n[keep].tolist(),
        },
    }


def summarise(out: dict[str, list[FocalRecord]], baselines: list[str], field_beliefs: bool = False) -> dict:
    summary = {}
    for name, records in out.items():
        row = {
            "profit": est(mean_ci([r.profit for r in records])),
            "goal_cards_held": est(mean_ci([r.goal_cards_held for r in records])),
            "trades_per_game": est(mean_ci([r.n_trades for r in records])),
            "field_profit": est(mean_ci([np.mean(r.field_profits) for r in records])),
        }
        for base in baselines:
            if base in out and base != name:
                row[f"vs_{base}"] = est(paired_diff_ci(records, out[base]))
        row["belief"] = belief_summary(records, lambda r: [r.focal_diagnostics])
        if field_beliefs:
            row["field_belief"] = belief_summary(records, lambda r: r.field_diagnostics)
            decoys = [r.focal_diagnostics.get("n_decoys", 0) for r in records]
            row["decoys_per_game"] = float(np.mean(decoys))
        summary[name] = row
    return summary


def correct_rate_by_hand_only(records: list[FocalRecord]) -> float:
    return float(np.mean([r.focal_diagnostics["goal_probs"][0].argmax() == r.goal_suit for r in records]))


def run(experiment: str, n_seeds: int) -> None:
    spec = agents()
    seeds = range(EVAL_SEED_START, EVAL_SEED_START + n_seeds)
    t0 = time.time()
    if experiment in ("headline", "bayes_field"):
        field_names = ["long_suit", "scarcity", "bayes_hand_only"] if experiment == "headline" else ["bayes"] * 3
        conditions = {name: spec[name] for name in HEADLINE_CONDITIONS}
        out = run_focal(conditions, [spec[n] for n in field_names], seeds, keep_diagnostics="focal")
        summary = summarise(out, baselines=["scarcity", "long_suit", "bayes_hand_only", "bayes"])
        summary["bayes"]["hand_only_accuracy_same_games"] = correct_rate_by_hand_only(out["bayes"])
    elif experiment == "leakage":
        field_names = ["bayes"] * 3
        conditions = {name: spec[name] for name in LEAKAGE_CONDITIONS}
        out = run_focal(conditions, [spec[n] for n in field_names], seeds, keep_diagnostics="all")
        summary = summarise(out, baselines=["bayes"], field_beliefs=True)
        for name in LEAKAGE_CONDITIONS[1:]:
            field_end = {r.seed: np.mean([d["goal_probs"][-1, r.goal_suit] for d in r.field_diagnostics]) for r in out["bayes"]}
            diffs = [np.mean([d["goal_probs"][-1, r.goal_suit] for d in r.field_diagnostics]) - field_end[r.seed] for r in out[name]]
            summary[name]["field_p_true_end_vs_bayes"] = est(mean_ci(diffs))
    else:
        raise ValueError(experiment)
    elapsed = time.time() - t0
    payload = {"experiment": experiment, "field": field_names, "n_seeds": n_seeds, "seconds": elapsed, "conditions": summary}
    path = write_json(f"{experiment}.json", payload)
    print(f"{experiment}: {n_seeds} seeds x {len(conditions)} conditions in {elapsed:.0f}s -> {path}")
    for name, row in summary.items():
        extras = "  ".join(f"{k}={row[k]['mean']:+.2f}[{row[k]['lo']:+.2f},{row[k]['hi']:+.2f}]" for k in row if k.startswith("vs_"))
        acc = row["belief"].get("accuracy_end", {}).get("mean") if row["belief"] else None
        acc_s = f" acc={acc:.3f}" if acc is not None else ""
        print(f"  {name:<18} profit {row['profit']['mean']:+7.2f} [{row['profit']['lo']:+.2f},{row['profit']['hi']:+.2f}]{acc_s}  {extras}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="*", default=["headline", "bayes_field", "leakage"])
    parser.add_argument("--seeds", type=int, default=3000)
    args = parser.parse_args()
    for experiment in args.experiments:
        run(experiment, args.seeds)


if __name__ == "__main__":
    main()
