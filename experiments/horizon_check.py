"""How much does the 120-turn trading horizon matter, and when does the market go quiet?

The four-minute round is discretised into turns, and 120 is an arbitrary choice. This
replays the headline comparison at three horizons and measures when trading actually
stops.
"""

from __future__ import annotations

import argparse

import numpy as np
from common import agents, write_json

from figgie.engine import Game
from figgie.protocol import TradeEvent
from figgie.tournament import agent_seed, focal_table, mean_ci, paired_diff_ci, run_focal

CONDITIONS = ("bayes", "bayes_hand_only", "bayes_flat_value", "bayes_mean_field")
HORIZONS = (60, 120, 240)


def last_trade_turns(spec, field, seeds: range, n_turns: int) -> tuple[np.ndarray, np.ndarray]:
    last, counts = [], []
    for seed in seeds:
        specs, _ = focal_table(spec["bayes"], field, seed)
        agents_ = [s.build(i, agent_seed(seed, i)) for i, s in enumerate(specs)]
        game = Game(agents_, seed=seed, n_turns=n_turns)
        result = game.run()
        trades = [e.turn for e in game.tape if isinstance(e, TradeEvent)]
        last.append(max(trades) if trades else -1)
        counts.append(result.n_trades)
    return np.array(last), np.array(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=800)
    args = parser.parse_args()
    spec = agents()
    field = [spec["long_suit"], spec["scarcity"], spec["bayes_hand_only"]]
    conditions = {n: spec[n] for n in CONDITIONS}

    summary = {"seeds": args.seeds, "horizons": {}}
    for turns in HORIZONS:
        out = run_focal(conditions, field, range(args.seeds), n_turns=turns)
        rows = {}
        for name in CONDITIONS:
            profit = mean_ci([r.profit for r in out[name]])
            row = {"profit": profit.mean}
            if name != "bayes":
                d = paired_diff_ci(out[name], out["bayes"])
                row["vs_bayes"] = {"mean": d.mean, "lo": d.lo, "hi": d.hi}
            rows[name] = row
        summary["horizons"][turns] = rows
        line = "  ".join(
            f"{n}={rows[n]['profit']:+.1f}" + ("" if n == "bayes" else f"({rows[n]['vs_bayes']['mean']:+.1f})")
            for n in CONDITIONS
        )
        print(f"n_turns={turns:<4} {line}")

    last, counts = last_trade_turns(spec, field, range(200), 120)
    summary["quiescence"] = {
        "median_last_trade_turn": float(np.median(last)),
        "p90_last_trade_turn": float(np.percentile(last, 90)),
        "share_trading_after_turn_60": float(np.mean(last > 60)),
        "trades_per_game": float(np.mean(counts)),
    }
    q = summary["quiescence"]
    print(f"median last trade: turn {q['median_last_trade_turn']:.0f} of 120 (p90 {q['p90_last_trade_turn']:.0f}); "
          f"{q['share_trading_after_turn_60']:.1%} of games trade after turn 60; "
          f"{q['trades_per_game']:.1f} trades per game")
    write_json("horizon_check.json", summary)


if __name__ == "__main__":
    main()
