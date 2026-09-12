from __future__ import annotations

import numpy as np

from ..deck import N_SUITS, PARTNER
from ..protocol import Observation
from .trader import ValuationTrader


class HeuristicTrader(ValuationTrader):
    """Picks one suit from its dealt hand, values it at a fixed price and everything else near zero.

    No inference beyond that choice and no awareness of inventory or order flow.
    """

    def __init__(self, seat: int, seed: int, goal_value: float = 12.0, other_value: float = 1.0, **kwargs):
        super().__init__(seat, seed, **kwargs)
        self.goal_value = goal_value
        self.other_value = other_value
        self._values: np.ndarray | None = None

    def target_suit(self, hand: tuple[int, ...]) -> int:
        raise NotImplementedError

    def _pick(self, hand, key) -> int:
        best = key(max(hand, key=key))
        return self.rng.choice([s for s in range(N_SUITS) if key(hand[s]) == best])

    def valuations(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        if self._values is None:
            self._values = np.full(N_SUITS, self.other_value)
            self._values[self.target_suit(obs.initial_hand)] = self.goal_value
        return self._values, self._values


class ScarcityHeuristic(HeuristicTrader):
    """Bids on whatever it was dealt fewest of."""

    name = "scarcity"

    def target_suit(self, hand):
        return self._pick(hand, key=lambda n: -n)


class LongSuitHeuristic(HeuristicTrader):
    """Bids on the same-colour partner of whatever it was dealt most of."""

    name = "long_suit"

    def target_suit(self, hand):
        return PARTNER[self._pick(hand, key=lambda n: n)]
