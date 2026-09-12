"""Paired-seed tournaments.

Every condition in a comparison plays the same seeds. The seed fixes the deck
configuration, the deal, the turn order and every seat's private RNG, so two
conditions that differ only in the focal agent face the identical deal and the
identical opponents. Paired differences strip out the enormous deal-to-deal
variance of Figgie.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .agents.base import AgentSpec
from .deck import N_PLAYERS
from .engine import DEFAULT_TURNS, Game, GameResult


def agent_seed(seed: int, seat: int) -> int:
    return seed * N_PLAYERS + seat


def play(seed: int, specs: Sequence[AgentSpec], n_turns: int = DEFAULT_TURNS) -> GameResult:
    agents = [spec.build(seat, agent_seed(seed, seat)) for seat, spec in enumerate(specs)]
    return Game(agents, seed=seed, n_turns=n_turns).run()


def focal_table(focal: AgentSpec, field: Sequence[AgentSpec], seed: int) -> tuple[list[AgentSpec], int]:
    """Seat the focal agent at seed % 4 (rotating seats across seeds), field in the rest."""
    seat = seed % N_PLAYERS
    specs = list(field)
    specs.insert(seat, focal)
    return specs, seat


@dataclass
class FocalRecord:
    condition: str
    seed: int
    seat: int
    profit: int
    goal_suit: int
    goal_count: int
    goal_cards_held: int
    n_trades: int
    field_profits: list[int]
    focal_diagnostics: dict = field(default_factory=dict)
    field_diagnostics: list[dict] = field(default_factory=list)


def _play_focal(args: tuple[str, AgentSpec, tuple[AgentSpec, ...], int, int, str | None]) -> FocalRecord:
    condition, focal, field_specs, seed, n_turns, keep_diagnostics = args
    specs, seat = focal_table(focal, field_specs, seed)
    result = play(seed, specs, n_turns)
    return FocalRecord(
        condition=condition,
        seed=seed,
        seat=seat,
        profit=result.profits[seat],
        goal_suit=result.config.goal_suit,
        goal_count=result.config.goal_count,
        goal_cards_held=result.goal_counts[seat],
        n_trades=result.n_trades,
        field_profits=[p for i, p in enumerate(result.profits) if i != seat],
        focal_diagnostics=result.diagnostics[seat] if keep_diagnostics else {},
        field_diagnostics=(
            [d for i, d in enumerate(result.diagnostics) if i != seat] if keep_diagnostics == "all" else []
        ),
    )


def run_focal(
    conditions: dict[str, AgentSpec],
    field: Sequence[AgentSpec],
    seeds: Iterable[int],
    n_turns: int = DEFAULT_TURNS,
    workers: int | None = None,
    keep_diagnostics: str | None = None,
) -> dict[str, list[FocalRecord]]:
    """Play every condition's focal agent against the same field on the same seeds.

    keep_diagnostics: None, "focal" (the focal agent's records) or "all" (field too).
    """
    seeds = list(seeds)
    jobs = [
        (name, spec, tuple(field), seed, n_turns, keep_diagnostics)
        for seed in seeds
        for name, spec in conditions.items()
    ]
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        records = [_play_focal(job) for job in jobs]
    else:
        with ProcessPoolExecutor(workers) as pool:
            records = list(pool.map(_play_focal, jobs, chunksize=max(1, len(jobs) // (workers * 8))))
    out: dict[str, list[FocalRecord]] = {name: [] for name in conditions}
    for r in records:
        out[r.condition].append(r)
    for rs in out.values():
        rs.sort(key=lambda r: r.seed)
    return out


# ---------------------------------------------------------------------------
# Statistics


@dataclass(frozen=True)
class Estimate:
    mean: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:+.2f} [{self.lo:+.2f}, {self.hi:+.2f}]"


def mean_ci(values: Sequence[float], z: float = 1.96) -> Estimate:
    x = np.asarray(values, dtype=np.float64)
    half = z * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return Estimate(float(x.mean()), float(x.mean() - half), float(x.mean() + half), len(x))


def paired_diff_ci(a: Sequence[FocalRecord], b: Sequence[FocalRecord]) -> Estimate:
    """Mean of (a - b) profit over matched seeds, with a normal-approximation 95% CI."""
    by_seed = {r.seed: r.profit for r in b}
    return mean_ci([r.profit - by_seed[r.seed] for r in a if r.seed in by_seed])
