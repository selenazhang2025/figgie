"""Tune execution parameters so no agent is a strawman.

The heuristics get their two value parameters swept; the Bayesian agent gets its
minimum offer price swept (the one execution knob that the heuristic sweep showed
matters: not giving away unwanted cards). Every variant plays as the focal agent
against a fixed mixed field on shared seeds, disjoint from the evaluation seeds.
The best setting per family is what the experiments use.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from figgie.agents import AgentSpec, BayesianTrader, LongSuitHeuristic, ScarcityHeuristic
from figgie.tournament import mean_ci, run_focal

TUNING_SEED_START = 50_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1500)
    parser.add_argument("--out", default="results/heuristic_tuning.json")
    args = parser.parse_args()

    field = [
        AgentSpec(LongSuitHeuristic, {"goal_value": 6, "other_value": 3}),
        AgentSpec(ScarcityHeuristic, {"goal_value": 6, "other_value": 3}),
        AgentSpec(BayesianTrader, {"use_order_flow": False}, label="bayes_hand_only"),
    ]
    variants: list[tuple[str, type, dict]] = []
    for cls, goal_value, other_value in itertools.product(
        (ScarcityHeuristic, LongSuitHeuristic), (3, 4, 6, 8, 12, 16), (2, 3, 4, 5)
    ):
        variants.append((cls.name, cls, {"goal_value": goal_value, "other_value": other_value}))
    for min_ask in (2, 3, 4, 6, 8, 10):
        variants.append(("bayes", BayesianTrader, {"min_ask": min_ask}))

    conditions = {
        f"{family}({', '.join(f'{k}={v}' for k, v in kwargs.items())})": AgentSpec(cls, kwargs, label=family)
        for family, cls, kwargs in variants
    }
    seeds = range(TUNING_SEED_START, TUNING_SEED_START + args.games)
    out = run_focal(conditions, field, seeds)

    rows = []
    for label, records in out.items():
        est = mean_ci([r.profit for r in records])
        rows.append({"label": label, "family": conditions[label].name, "kwargs": conditions[label].kwargs,
                     "mean": est.mean, "lo": est.lo, "hi": est.hi, "n": est.n})
    rows.sort(key=lambda r: -r["mean"])
    for r in rows:
        print(f"{r['label']:<42} {r['mean']:+7.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
    best = {family: next(r for r in rows if r["family"] == family) for family in ("scarcity", "long_suit", "bayes")}
    print("best:", {k: v["kwargs"] for k, v in best.items()})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"games": args.games, "field": [s.name for s in field], "rows": rows, "best": best}, indent=2))


if __name__ == "__main__":
    main()
