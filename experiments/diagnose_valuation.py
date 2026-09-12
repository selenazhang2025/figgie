"""Why does a flat card valuation beat the inventory-aware one against Bayesian opponents?

Plays the full and the flat-value Bayesian agents as the focal player against three full
Bayesian opponents on the same deals, and breaks the focal player's trading down into
goal-suit buys and sells, other suits, and the majority bonus.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from common import agents, write_json

from figgie.engine import Game
from figgie.protocol import TradeEvent
from figgie.tournament import agent_seed, focal_table

CONDITIONS = ("bayes", "bayes_flat_value")
FIELD = "bayes"


def one(args: tuple[int, str]) -> tuple[str, dict]:
    seed, focal_name = args
    spec = agents()
    specs, seat = focal_table(spec[focal_name], [spec[FIELD]] * 3, seed)
    game = Game([s.build(i, agent_seed(seed, i)) for i, s in enumerate(specs)], seed=seed)
    result = game.run()
    goal = result.config.goal_suit
    m = {"bought": 0, "buy_cost": 0, "sold": 0, "sell_revenue": 0, "other_suits_net": 0}
    for ev in game.tape:
        if not isinstance(ev, TradeEvent):
            continue
        if ev.buyer == seat:
            if ev.suit == goal:
                m["bought"] += 1
                m["buy_cost"] += ev.price
            else:
                m["other_suits_net"] -= ev.price
        if ev.seller == seat:
            if ev.suit == goal:
                m["sold"] += 1
                m["sell_revenue"] += ev.price
            else:
                m["other_suits_net"] += ev.price
    held = result.goal_counts[seat]
    m.update(profit=result.profits[seat], goal_held=held, bonus=result.payouts[seat] - 10 * held)
    return focal_name, m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=300)
    args = parser.parse_args()
    jobs = [(seed, name) for name in CONDITIONS for seed in range(args.seeds)]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(one, jobs, chunksize=10))

    summary = {}
    for name in CONDITIONS:
        ms = [m for n, m in rows if n == name]
        total = {k: float(np.sum([m[k] for m in ms])) for k in ms[0]}
        summary[name] = {
            "profit": total["profit"] / len(ms),
            "goal_bought_per_game": total["bought"] / len(ms),
            "goal_buy_price": total["buy_cost"] / max(total["bought"], 1),
            "goal_sold_per_game": total["sold"] / len(ms),
            "goal_sell_price": total["sell_revenue"] / max(total["sold"], 1),
            "other_suits_net_per_game": total["other_suits_net"] / len(ms),
            "goal_held": total["goal_held"] / len(ms),
            "bonus_won_rate": float(np.mean([m["bonus"] > 0 for m in ms])),
        }
        s = summary[name]
        print(f"{name:<17} profit {s['profit']:+6.1f} | goal bought {s['goal_bought_per_game']:.2f} @ ${s['goal_buy_price']:.1f}"
              f"  sold {s['goal_sold_per_game']:.2f} @ ${s['goal_sell_price']:.1f} | other suits {s['other_suits_net_per_game']:+.1f}"
              f" | bonus won {s['bonus_won_rate']:.0%}")
    write_json("valuation_mechanism.json", {"seeds": args.seeds, "field": [FIELD] * 3, "conditions": summary})


if __name__ == "__main__":
    main()
