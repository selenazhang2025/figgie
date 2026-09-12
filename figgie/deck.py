"""Suits, deck configurations and dealing.

A Figgie deck has 40 cards: one suit with 12, two with 10, one with 8. The goal
suit is the same-colour partner of the 12-card suit, so it always has 8 or 10.
There are exactly twelve possible configurations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cache
from itertools import product

import numpy as np

N_SUITS = 4
N_PLAYERS = 4
DECK_SIZE = 40
HAND_SIZE = 10

STARTING_CHIPS = 350
ANTE = 50
POT = ANTE * N_PLAYERS  # 200
CARD_PAYOUT = 10

SPADES, CLUBS, HEARTS, DIAMONDS = range(N_SUITS)
SUIT_NAMES = ("spades", "clubs", "hearts", "diamonds")
SUIT_SYMBOLS = ("♠", "♣", "♥", "♦")
# Same-colour partner: spades <-> clubs (black), hearts <-> diamonds (red).
PARTNER = (CLUBS, SPADES, DIAMONDS, HEARTS)


@dataclass(frozen=True, slots=True)
class DeckConfig:
    counts: tuple[int, int, int, int]

    @property
    def long_suit(self) -> int:
        return self.counts.index(12)

    @property
    def goal_suit(self) -> int:
        return PARTNER[self.long_suit]

    @property
    def goal_count(self) -> int:
        return self.counts[self.goal_suit]

    @property
    def bonus(self) -> int:
        return POT - CARD_PAYOUT * self.goal_count

    def __str__(self) -> str:
        parts = " ".join(f"{SUIT_SYMBOLS[s]}{n}" for s, n in enumerate(self.counts))
        return f"[{parts} goal={SUIT_SYMBOLS[self.goal_suit]}]"


@cache
def all_configs() -> tuple[DeckConfig, ...]:
    configs = []
    for long_suit in range(N_SUITS):
        goal = PARTNER[long_suit]
        others = [s for s in range(N_SUITS) if s not in (long_suit, goal)]
        # Goal has 8: both off-colour suits have 10.
        # Goal has 10: one off-colour suit has 10, the other 8 (two ways).
        for goal_n, other_ns in ((8, (10, 10)), (10, (10, 8)), (10, (8, 10))):
            counts = [0] * N_SUITS
            counts[long_suit] = 12
            counts[goal] = goal_n
            counts[others[0]], counts[others[1]] = other_ns
            configs.append(DeckConfig(tuple(counts)))
    return tuple(configs)


N_CONFIGS = len(all_configs())
CONFIG_COUNTS = np.array([c.counts for c in all_configs()], dtype=np.int64)  # (12, 4)
CONFIG_GOAL = np.array([c.goal_suit for c in all_configs()], dtype=np.int64)  # (12,)
CONFIG_GOAL_COUNT = np.array([c.goal_count for c in all_configs()], dtype=np.int64)
CONFIG_BONUS = np.array([c.bonus for c in all_configs()], dtype=np.float64)


@cache
def _hands() -> np.ndarray:
    rows = [h for h in product(range(HAND_SIZE + 1), repeat=N_SUITS) if sum(h) == HAND_SIZE]
    return np.array(rows, dtype=np.int64)


HANDS = _hands()  # (286, 4): every suit-count composition of a 10-card hand
N_HANDS = len(HANDS)
HAND_INDEX = {tuple(int(x) for x in h): i for i, h in enumerate(HANDS)}


def sample_config(rng: random.Random) -> DeckConfig:
    return rng.choice(all_configs())


def deal(config: DeckConfig, rng: random.Random) -> list[list[int]]:
    """Shuffle the physical deck and deal ten cards to each player, as suit counts."""
    deck = [s for s, n in enumerate(config.counts) for _ in range(n)]
    rng.shuffle(deck)
    hands = []
    for p in range(N_PLAYERS):
        counts = [0] * N_SUITS
        for s in deck[p * HAND_SIZE : (p + 1) * HAND_SIZE]:
            counts[s] += 1
        hands.append(counts)
    return hands
