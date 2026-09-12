"""Tune execution parameters so that no agent is a strawman.

Every family is tuned over the same knobs by coordinate search: sweep one parameter
with the others fixed, keep the best, move to the next, and repeat. Tuning a parameter
for one family and not another would hand that family an unearned edge, which is why
the minimum offer price is swept for the heuristics too and not only for the Bayesian
agent.

Each variant plays as the focal agent against a fixed field on shared seeds, disjoint
from the evaluation seeds. The best setting per family is what the experiments use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from figgie.agents import AgentSpec, BayesianTrader, LongSuitHeuristic, ScarcityHeuristic
from figgie.tournament import mean_ci, run_focal

TUNING_SEED_START = 50_000
ROUNDS = 2

# family -> (class, starting parameters)
FAMILIES = {
    "scarcity": (ScarcityHeuristic, {"goal_value": 6, "other_value": 3, "min_ask": 2}),
    "long_suit": (LongSuitHeuristic, {"goal_value": 6, "other_value": 3, "min_ask": 2}),
    "bayes": (BayesianTrader, {"min_ask": 2}),
}
SWEEPS = {
    "goal_value": (3, 4, 6, 8, 12, 16),
    "other_value": (2, 3, 4, 5, 6, 7),
    "min_ask": (2, 4, 6, 10, 15, 20, 25, 35),
}


def label_of(family: str, kwargs: dict) -> str:
    return f"{family}({', '.join(f'{k}={v}' for k, v in sorted(kwargs.items()))})"


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
    seeds = range(TUNING_SEED_START, TUNING_SEED_START + args.games)
    current = {family: dict(start) for family, (_, start) in FAMILIES.items()}
    scored: dict[str, dict] = {}  # label -> row, also the cache across rounds

    for round_ in range(ROUNDS):
        for param, values in SWEEPS.items():
            conditions, wanted = {}, {}
            for family, (cls, _) in FAMILIES.items():
                if param not in current[family]:
                    continue
                for value in values:
                    kwargs = {**current[family], param: value}
                    label = label_of(family, kwargs)
                    wanted.setdefault(family, []).append(label)
                    if label not in scored:
                        conditions[label] = AgentSpec(cls, kwargs, label=family)
            if conditions:
                for label, records in run_focal(conditions, field, seeds).items():
                    est = mean_ci([r.profit for r in records])
                    scored[label] = {"label": label, "family": conditions[label].name,
                                     "kwargs": conditions[label].kwargs, "round": round_, "param": param,
                                     "mean": est.mean, "lo": est.lo, "hi": est.hi, "n": est.n}
            for family, labels in wanted.items():
                best = max(labels, key=lambda label: scored[label]["mean"])
                current[family] = dict(scored[best]["kwargs"])
                print(f"round {round_} {param:<12} {family:<10} -> {scored[best]['kwargs']}  {scored[best]['mean']:+.2f}")

    rows = sorted(scored.values(), key=lambda r: -r["mean"])
    best = {family: scored[label_of(family, current[family])] for family in FAMILIES}
    print("\ntop settings per family:")
    for family, row in best.items():
        print(f"  {family:<10} {row['kwargs']}  {row['mean']:+.2f} [{row['lo']:+.2f}, {row['hi']:+.2f}]")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"games": args.games, "field": [s.name for s in field], "rounds": ROUNDS,
         "sweeps": {k: list(v) for k, v in SWEEPS.items()}, "rows": rows, "best": best}, indent=2))


if __name__ == "__main__":
    main()
