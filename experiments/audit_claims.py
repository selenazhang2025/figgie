"""Checks on the simulation itself, for claims the experiments don't already produce.

  symmetry    four identical agents: every seat must average zero, every game zero-sum
  ties        how often the majority for the goal suit is shared
  calibration when the posterior says 80%, is it right 80% of the time?
"""

from __future__ import annotations

import argparse
import platform
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from common import agents, write_json

from figgie.tournament import focal_table, play


def _symmetry_game(seed: int) -> tuple[list[int], int]:
    spec = agents()
    result = play(seed, [spec["bayes"]] * 4)
    counts = sorted(result.goal_counts)
    return result.profits, int(counts[-1] == counts[-2])


def _calibration_game(seed: int) -> tuple[float, int]:
    spec = agents()
    specs, seat = focal_table(spec["bayes"], [spec["long_suit"], spec["scarcity"], spec["bayes_hand_only"]], seed)
    result = play(seed, specs)
    probs = result.diagnostics[seat]["goal_probs"][-1]
    return float(probs.max()), int(probs.argmax() == result.config.goal_suit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=800)
    args = parser.parse_args()

    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(_symmetry_game, range(args.games), chunksize=10))
        calib = list(pool.map(_calibration_game, range(args.games), chunksize=10))

    profits = np.array([r[0] for r in rows], dtype=float)
    half = 1.96 * profits.std(axis=0, ddof=1) / np.sqrt(len(profits))
    seat_means = profits.mean(axis=0)
    tie_rate = float(np.mean([r[1] for r in rows]))

    confidence = np.array([c[0] for c in calib])
    correct = np.array([c[1] for c in calib], dtype=float)
    bins = [(0.25, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]
    reliability = []
    for lo, hi in bins:
        m = (confidence >= lo) & (confidence < hi)
        if m.sum() >= 10:
            reliability.append({"lo": lo, "hi": hi, "says": float(confidence[m].mean()),
                                "right": float(correct[m].mean()), "n": int(m.sum())})

    payload = {
        "games": args.games,
        # Games are a deterministic function of their seed, for a given library version.
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform()},
        "symmetry": {"seat_mean_profit": seat_means.tolist(), "ci_half_width": half.tolist(),
                     "every_game_zero_sum": bool(np.all(profits.sum(axis=1) == 0))},
        "tie_rate": tie_rate,
        "calibration": {"mean_confidence": float(confidence.mean()), "accuracy": float(correct.mean()),
                        "overconfidence": float(confidence.mean() - correct.mean()), "reliability": reliability},
    }
    print(f"symmetry over {args.games} games with four identical agents:")
    for seat in range(4):
        print(f"   seat {seat}: {seat_means[seat]:+6.2f} +/- {half[seat]:.2f}")
    print(f"   every game zero-sum: {payload['symmetry']['every_game_zero_sum']}")
    print(f"ties for most goal cards: {tie_rate:.1%}")
    c = payload["calibration"]
    print(f"posterior calibration: says {c['mean_confidence']:.3f}, right {c['accuracy']:.3f} "
          f"(overconfident by {c['overconfidence']:+.3f})")
    for r in reliability:
        print(f"   {r['lo']:.2f}-{r['hi']:.2f}: says {r['says']:.3f}, right {r['right']:.3f} (n={r['n']})")
    write_json("audit.json", payload)


if __name__ == "__main__":
    main()
