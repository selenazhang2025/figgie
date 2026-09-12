"""What one more goal-suit card is worth to you, given what you already hold.

Holding k goal cards pays 10k plus the bonus times your expected share of it:
1 if you hold strictly the most, 1/(t+1) if you tie with t opponents, else 0.
Buying one card adds $10 and moves your expected bonus share, and that second
term depends heavily on k. It is small with one card, largest near the
threshold where one card flips you into the lead, and small again once you are
safely ahead. So the marginal value of a card is hump-shaped in inventory.

Opponents' holdings enter through three independent per-opponent distributions
q_j(x), conditioned on the known total (all goal cards not in your hand are in
theirs). Buying a card removes it from an opponent chosen in proportion to their
holdings; selling hands it to an opponent chosen uniformly.
"""

from __future__ import annotations

from functools import cache
from math import comb

import numpy as np

from .deck import CARD_PAYOUT, CONFIG_BONUS, CONFIG_GOAL, CONFIG_GOAL_COUNT, N_SUITS, POT

MAX_HELD = 10

# Tracks how often a total had no support in the belief, and the largest share of the
# posterior that was ever affected. Configurations the agent has already ruled out cost
# nothing; "weight" is what matters, and the tests assert it stays negligible.
NO_SUPPORT = {"count": 0, "weight": 0.0}
_V = MAX_HELD + 1
_N_OPP = 3

_grid = np.meshgrid(*(np.arange(_V),) * _N_OPP, indexing="ij")
CELLS = np.stack([g.ravel() for g in _grid], axis=1)  # (1331, 3): opponents' goal counts
CELL_SUM = CELLS.sum(axis=1)
_CELL_MAX = CELLS.max(axis=1)
_CELL_TIES = (CELLS == _CELL_MAX[:, None]).sum(axis=1)
_K = np.arange(MAX_HELD + 2)[:, None]
# SHARE[k, cell]: my share of the bonus holding k against opponents at `cell`.
SHARE = np.where(_K > _CELL_MAX, 1.0, np.where(_K == _CELL_MAX, 1.0 / (_CELL_TIES + 1), 0.0))

# Prior per-opponent distribution at the deal: q(x) ∝ C(10, x). Conditioned on the
# total this is exactly the multivariate hypergeometric split of the remaining cards.
PRIOR_Q = np.array([comb(MAX_HELD, x) for x in range(_V)], dtype=np.float64)
PRIOR_Q /= PRIOR_Q.sum()


def _flat_index(cells: np.ndarray) -> np.ndarray:
    return (cells[:, 0] * _V + cells[:, 1]) * _V + cells[:, 2]


def _share_after_shift(k: int, delta: int) -> np.ndarray:
    """(3, 1331): my share holding k after opponent j's count changes by delta."""
    out = np.zeros((_N_OPP, len(CELLS)))
    for j in range(_N_OPP):
        moved = CELLS.copy()
        moved[:, j] += delta
        ok = (moved[:, j] >= 0) & (moved[:, j] <= MAX_HELD)
        out[j] = np.where(ok, SHARE[k][_flat_index(np.clip(moved, 0, MAX_HELD))], 0.0)
    return out


@cache
def share_tables(n_goal: int, conditioned: bool = True) -> np.ndarray:
    """(11, 4, 1331) linear functionals of the opponents' joint, indexed by my holding k.

    Contracting row [k, t] with an (unnormalised) joint over cells gives:
      t=0: probability mass consistent with holding k (the normaliser)
      t=1: expected share now
      t=2: expected share after buying one card
      t=3: expected share after selling one card

    `conditioned` restricts to holdings that add up to the goal cards not in my hand.
    A belief that ignores order flow can rule that total out while still believing the
    configuration, so the unconditioned tables are the fallback for those cases.
    """
    tables = np.zeros((_V, 4, len(CELLS)))
    for k in range(min(n_goal, MAX_HELD) + 1):
        others = n_goal - k
        mask = (CELL_SUM == others).astype(np.float64) if conditioned else np.ones(len(CELLS))
        held_by_others = max(others, 1) if conditioned else np.maximum(CELL_SUM, 1)
        tables[k, 0] = mask
        tables[k, 1] = mask * SHARE[k]
        if k < n_goal:
            seller_weight = CELLS.T / held_by_others  # (3, 1331)
            tables[k, 2] = mask * (seller_weight * _share_after_shift(k + 1, -1)).sum(axis=0)
        if k > 0:
            tables[k, 3] = mask * _share_after_shift(k - 1, +1).mean(axis=0)
    return tables


def joint(q: np.ndarray) -> np.ndarray:
    """Independent product of three opponents' count distributions, flattened: (..., 1331)."""
    j = q[..., 0, :, None, None] * q[..., 1, None, :, None] * q[..., 2, None, None, :]
    return j.reshape(*q.shape[:-2], len(CELLS))


def expected_shares(
    q: np.ndarray, n_goal: np.ndarray, k: np.ndarray, probs: np.ndarray | None = None
) -> np.ndarray:
    """Batched (..., 3) array of [share now, share after buying, share after selling].

    `probs` is only used to record how much posterior weight hits the fallback below.
    """
    n_goal = np.asarray(n_goal)
    k = np.minimum(np.asarray(k), n_goal)
    tables = np.stack([share_tables(int(n))[int(kk)] for n, kk in zip(n_goal.ravel(), k.ravel())])
    tables = tables.reshape(*n_goal.shape, 4, len(CELLS))
    moments = np.einsum("...n,...tn->...t", joint(q), tables)
    mass = moments[..., 0]
    empty = mass <= 1e-300
    if np.any(empty):
        # This belief holds no opponent holdings that add up to the cards left outside my
        # hand, which happens when a belief that ignores order flow still believes in a
        # configuration the trades have ruled out. Keep what it does think opponents hold
        # and drop only the total, rather than falling back on the deal and discarding it.
        NO_SUPPORT["count"] += int(np.sum(empty))
        if probs is not None:
            NO_SUPPORT["weight"] = max(NO_SUPPORT["weight"], float(np.asarray(probs)[empty].sum()))
        loose = np.stack([share_tables(int(n), False)[int(kk)] for n, kk in zip(n_goal.ravel(), k.ravel())])
        loose = loose.reshape(*n_goal.shape, 4, len(CELLS))
        moments = np.where(empty[..., None], np.einsum("...n,...tn->...t", joint(q), loose), moments)
        mass = moments[..., 0]
        if np.any(mass <= 1e-300):  # the belief has no opponent holdings at all: use the deal
            prior = np.einsum("n,...tn->...t", joint(np.broadcast_to(PRIOR_Q, (_N_OPP, _V))), tables)
            moments = np.where((mass <= 1e-300)[..., None], prior, moments)
            mass = moments[..., 0]
    return moments[..., 1:] / mass[..., None]


def card_values(
    config_probs: np.ndarray, q: np.ndarray, hand, inventory_aware: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Per-suit value of buying one card and of giving one up.

    config_probs: (12,) posterior. q: (12, 3, 11) opponents' goal-count distributions
    under each config. Returns (buy_value, sell_value), each (4,), in chips.

    A card of suit s is only worth anything in configurations where s is the goal, so
    the bid ceiling is sum over those configs of P(config) * marginal value there.
    With inventory_aware=False every goal card is valued at the flat average pot/n.
    """
    hand = np.asarray(hand, dtype=np.int64)
    k = hand[CONFIG_GOAL]
    if inventory_aware:
        s_now, s_buy, s_sell = np.moveaxis(expected_shares(q, CONFIG_GOAL_COUNT, k, config_probs), -1, 0)
        mv_buy = CARD_PAYOUT + CONFIG_BONUS * (s_buy - s_now)
        mv_sell = CARD_PAYOUT + CONFIG_BONUS * (s_now - s_sell)
    else:
        mv_buy = mv_sell = POT / CONFIG_GOAL_COUNT
    mv_buy = np.where(k < CONFIG_GOAL_COUNT, mv_buy, 0.0)
    mv_sell = np.where(k > 0, mv_sell, 0.0)
    buy_value = np.bincount(CONFIG_GOAL, weights=config_probs * mv_buy, minlength=N_SUITS)
    sell_value = np.bincount(CONFIG_GOAL, weights=config_probs * mv_sell, minlength=N_SUITS)
    return buy_value, sell_value


def marginal_value_curve(n_goal: int, q: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Marginal value of a goal card against holdings k = 0..n_goal-1, for a known goal suit.

    With q=None opponents' holdings follow the deal (hypergeometric). Returns k, the
    expected bonus share at k, expected payout at k, and the buy-side marginal value
    split into the flat $10 and the bonus term.
    """
    bonus = POT - CARD_PAYOUT * n_goal
    q = np.broadcast_to(PRIOR_Q if q is None else q, (_N_OPP, _V))
    ks = np.arange(n_goal + 1)
    shares = np.stack([expected_shares(q, np.array(n_goal), np.array(k)) for k in ks])
    s_now, s_buy = shares[:, 0], shares[:, 1]
    return {
        "k": ks[:-1],
        "share": s_now[:-1],
        "payout": CARD_PAYOUT * ks[:-1] + bonus * s_now[:-1],
        "bonus_term": bonus * (s_buy - s_now)[:-1],
        "marginal_value": CARD_PAYOUT + bonus * (s_buy - s_now)[:-1],
    }
