"""Fit the level-0 opponent model by maximum likelihood on simulated games.

Fitting scores every placement on the tape against the placer's *true* dealt hand,
which only the harness knows after a game ends. The winning parameters become fixed
constants inside the agents; no agent sees hidden state while playing.

Reports the average log-likelihood per placement in bits, relative to a model that
picks suits uniformly (among suits it could legally offer).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from figgie.agents import AgentSpec, BayesianTrader, LongSuitHeuristic, ScarcityHeuristic
from figgie.deck import HAND_INDEX
from figgie.engine import Game
from figgie.inference import BeliefState, Level0Model
from figgie.protocol import OrderPlaced
from figgie.tournament import agent_seed

POPULATION = [
    AgentSpec(BayesianTrader, label="bayes"),
    AgentSpec(BayesianTrader, {"use_order_flow": False}, label="bayes_hand_only"),
    AgentSpec(LongSuitHeuristic),
    AgentSpec(ScarcityHeuristic),
]

_GAMES: list = []


def simulate(seed: int):
    shift = seed % 4
    specs = POPULATION[shift:] + POPULATION[:shift]
    agents = [spec.build(seat, agent_seed(seed, seat)) for seat, spec in enumerate(specs)]
    game = Game(agents, seed=seed)
    result = game.run()
    return tuple(game.tape), result.initial_hands


def _init(games):
    global _GAMES
    _GAMES = games


def score(model: Level0Model) -> tuple[float, int]:
    total, n = 0.0, 0
    for tape, hands in _GAMES:
        observer = BeliefState(-1, (0, 0, 0, 0), model)  # seat -1: every player is an "opponent"
        observer.update(tape)
        total += sum(observer.behaviour[p, HAND_INDEX[tuple(h)]] for p, h in enumerate(hands))
        n += sum(isinstance(ev, OrderPlaced) for ev in tape)
    return total, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--out", default="results/opponent_model_fit.json")
    args = parser.parse_args()

    workers = os.cpu_count() or 1
    with ProcessPoolExecutor(workers) as pool:
        games = list(pool.map(simulate, range(10_000, 10_000 + args.games), chunksize=8))

    grid = [Level0Model("posterior", b, e) for b, e in itertools.product([1, 2, 4, 6, 8, 12, 16], [0.1, 0.2, 0.3, 0.5, 0.7])]
    grid += [Level0Model("scarcity", b, e) for b, e in itertools.product([0.1, 0.25, 0.5, 1, 2], [0.1, 0.2, 0.3, 0.5, 0.7])]
    baseline = Level0Model("posterior", 0.0, 1.0)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(games,)) as pool:
        scores = list(pool.map(score, [baseline, *grid]))
    base_total, n = scores[0]
    rows = []
    for model, (total, _) in zip(grid, scores[1:]):
        bits = (total - base_total) / n / np.log(2)
        rows.append({"kind": model.kind, "beta": model.beta, "epsilon": model.epsilon, "bits_per_placement": bits})
    rows.sort(key=lambda r: -r["bits_per_placement"])
    for r in rows[:8]:
        print(f"{r['kind']:>9}  beta={r['beta']:<5} eps={r['epsilon']:<4} {r['bits_per_placement']:+.4f} bits/placement")
    for kind in ("posterior", "scarcity"):
        best = next(r for r in rows if r["kind"] == kind)
        print(f"best {kind}: {best}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"placements": n, "games": args.games, "grid": rows}, indent=2))


if __name__ == "__main__":
    main()
