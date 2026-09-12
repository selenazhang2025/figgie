"""Exact Bayesian inference over the twelve deck configurations.

Two sources of evidence:

1. Your own dealt hand. Ten cards drawn without replacement from a deck whose
   composition depends on the configuration, so P(hand | config) is a
   multivariate hypergeometric. Exact, twelve numbers.

2. Order flow. Each opponent's placements are noisy signals about their dealt
   hand, and their hand is a signal about the deck. A level-0 opponent model
   turns each placement into a likelihood over the 286 possible dealt hands.
   Marginalising over the *joint* deal of the 30 cards you can't see gives a
   likelihood per configuration.

   The joint sum looks like 286^2 terms per configuration, but the deal
   probability factorises: P(h_B, h_C, h_D | config) is proportional to
   prod_s r_s! / (h_Bs! h_Cs! h_Ds!), where r is what's left after your hand. So
   the sum is a convolution over suit-count compositions: pair up two opponents'
   hands into a 20-card composition with one bincount, then dot with the third.
   `exact_config_posterior` does the brute-force enumeration and the tests check
   the two agree.

Trades also carry hard, model-free constraints: nobody can sell a card they do
not hold, so an opponent who has sold three spades was dealt at least three
minus whatever spades they bought. These enter as zero-likelihood hands, and the
joint sum enforces them across opponents (two players can't each have been dealt
six of a ten-card suit).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from .deck import (
    CONFIG_COUNTS,
    CONFIG_GOAL,
    DECK_SIZE,
    HAND_SIZE,
    HANDS,
    N_CONFIGS,
    N_HANDS,
    N_PLAYERS,
    N_SUITS,
)
from .market import Side
from .protocol import Event, OrderCancelled, OrderPlaced, TradeEvent

NEG_INF = -np.inf
MAX_GOAL_HELD = 10
_LOG_FACT = np.array([math.lgamma(n + 1) for n in range(DECK_SIZE + 1)])
GOAL_ONEHOT = np.eye(N_SUITS)[CONFIG_GOAL]  # (12, 4)


def log_comb(n, k) -> np.ndarray:
    """log C(n, k), -inf where the combination is impossible (k < 0 or k > n)."""
    n = np.asarray(n)
    k = np.asarray(k)
    valid = (k >= 0) & (n >= k)
    nn = np.clip(n, 0, DECK_SIZE)
    kk = np.clip(k, 0, DECK_SIZE)
    out = _LOG_FACT[nn] - _LOG_FACT[kk] - _LOG_FACT[np.clip(nn - kk, 0, DECK_SIZE)]
    return np.where(valid, out, NEG_INF)


def logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    shift = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - shift), axis=axis, keepdims=True)) + shift
    return np.squeeze(out, axis=axis)


def normalise_log(log_w: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax along an axis; slices with no mass come back as zeros rather than NaN."""
    z = np.expand_dims(logsumexp(log_w, axis=axis), axis)
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.exp(log_w - z), nan=0.0)


# ---------------------------------------------------------------------------
# Hand evidence


def hand_log_likelihood(hand) -> np.ndarray:
    """log P(dealt hand | config) for all 12 configs. Accepts (4,) or batched (..., 4)."""
    h = np.asarray(hand, dtype=np.int64)
    return log_comb(CONFIG_COUNTS, h[..., None, :]).sum(axis=-1) - log_comb(DECK_SIZE, HAND_SIZE)


def hand_posterior(hand) -> np.ndarray:
    """P(config | dealt hand) under the uniform prior."""
    return normalise_log(hand_log_likelihood(hand))


def goal_marginal(config_probs: np.ndarray) -> np.ndarray:
    """Collapse a distribution over configurations to P(goal suit = s)."""
    return config_probs @ GOAL_ONEHOT


# P(goal = s | dealt hand h) for each of the 286 hands: what a hand-only player believes.
HAND_ONLY_GOAL = goal_marginal(hand_posterior(HANDS))  # (286, 4)


def opponent_hand_log_prior(my_hand) -> np.ndarray:
    """log P(one opponent was dealt h | config, my dealt hand), shape (12, 286).

    Marginally, each opponent's ten cards are a hypergeometric draw from the 30 I can't see.
    """
    remaining = CONFIG_COUNTS - np.asarray(my_hand, dtype=np.int64)  # (12, 4)
    rest = DECK_SIZE - HAND_SIZE
    return log_comb(remaining[:, None, :], HANDS[None, :, :]).sum(axis=-1) - log_comb(rest, HAND_SIZE)


# ---------------------------------------------------------------------------
# Joint deal of the unseen cards, as a convolution over suit-count compositions

_BASE = 2 * HAND_SIZE + 1


def _encode(counts: np.ndarray) -> np.ndarray:
    c = np.clip(counts, 0, _BASE - 1)
    return ((c[..., 0] * _BASE + c[..., 1]) * _BASE + c[..., 2]) * _BASE + c[..., 3]


_PAIRS = np.array([u for u in product(range(_BASE), repeat=N_SUITS) if sum(u) == 2 * HAND_SIZE])
N_PAIRS = len(_PAIRS)  # 1771 compositions of 20 cards into 4 suits
_PAIR_LOOKUP = np.full(_BASE**N_SUITS, -1, dtype=np.int64)
_PAIR_LOOKUP[_encode(_PAIRS)] = np.arange(N_PAIRS)
PAIR_OF = _PAIR_LOOKUP[_encode(HANDS[:, None, :] + HANDS[None, :, :])]  # (286, 286)
LOG_HAND_FACT = _LOG_FACT[HANDS].sum(axis=1)  # sum_s log h_s!


def rest_index(my_hand) -> tuple[np.ndarray, np.ndarray]:
    """For each (config, third opponent's hand): index of the 20-card composition the other
    two must jointly hold, or -1 if impossible. Also sum_s log r_s! per config."""
    remaining = CONFIG_COUNTS - np.asarray(my_hand, dtype=np.int64)  # (12, 4)
    rest = remaining[:, None, :] - HANDS[None, :, :]  # (12, 286, 4)
    index = np.where((rest >= 0).all(axis=-1), _PAIR_LOOKUP[_encode(rest)], -1)
    ok = (remaining >= 0).all(axis=1)
    log_rest_fact = np.where(ok, _LOG_FACT[np.clip(remaining, 0, DECK_SIZE)].sum(axis=1), NEG_INF)
    return index, log_rest_fact


def _pair_sums(ga: np.ndarray, gb: np.ndarray) -> np.ndarray:
    ia, ib = np.flatnonzero(ga), np.flatnonzero(gb)
    return np.bincount(
        PAIR_OF[np.ix_(ia, ib)].ravel(), weights=np.outer(ga[ia], gb[ib]).ravel(), minlength=N_PAIRS
    )


def joint_hand_posterior(
    opp_log_lik: np.ndarray, index: np.ndarray, log_rest_fact: np.ndarray, prune: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Exact log P(flow | config) up to a constant, and P(opponent j dealt h | config, flow).

    opp_log_lik: (3, 286) log P(flow of opponent j | j's dealt hand).
    Returns (log_evidence (12,), weights (3, 12, 286)).
    """
    log_g = opp_log_lik - LOG_HAND_FACT
    shift = np.max(log_g, axis=1, keepdims=True)
    if not np.isfinite(shift).all():
        return np.full(N_CONFIGS, NEG_INF), np.zeros((3, N_CONFIGS, N_HANDS))
    g = np.exp(log_g - shift)
    g[g < prune] = 0.0
    valid = index >= 0
    safe = np.where(valid, index, 0)
    marginals = np.empty((3, N_CONFIGS, N_HANDS))
    for j in range(3):
        a, b = (k for k in range(3) if k != j)
        pair = _pair_sums(g[a], g[b])
        marginals[j] = g[j][None, :] * np.where(valid, pair[safe], 0.0)
    z = marginals[2].sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_evidence = np.log(z) + log_rest_fact
        weights = np.nan_to_num(marginals / marginals.sum(axis=-1, keepdims=True), nan=0.0)
    return log_evidence, weights


# ---------------------------------------------------------------------------
# Opponent model


@dataclass(frozen=True)
class Level0Model:
    """Level-0 opponent: chooses which suit to bid on or offer by a softmax over a per-suit
    score computed from its own hand alone. It does not reason about anyone else's orders.

    kind="posterior": score_s = P(goal = s | its dealt hand). Bids favour likely goal
        suits, offers favour unlikely ones.
    kind="scarcity":  score_s = -(cards of s it holds now). Bids favour suits it is
        short, offers suits it is long: the "bid what you hold least of" player.

    Price sharpens the choice: a bid at twice the reference price is treated as twice
    as deliberate, an offer at a giveaway price as a strong "I don't want this". A
    fraction `epsilon` of placements are uniform noise, which keeps the likelihood
    robust to players the model describes badly.

    Defaults are the maximum-likelihood fit from experiments/calibrate_opponent_model.py.
    """

    kind: str = "posterior"
    beta: float = 4.0
    epsilon: float = 0.2
    price_ref: float = 10.0

    def scores(self, current: np.ndarray) -> np.ndarray:
        if self.kind == "posterior":
            return HAND_ONLY_GOAL
        if self.kind == "scarcity":
            return -current.astype(np.float64)
        raise ValueError(self.kind)

    def log_likelihood(
        self, side: Side, suit: int, price: int, current: np.ndarray, open_sells: np.ndarray
    ) -> np.ndarray:
        """log P(chose `suit` | side, price, dealt hand) for each of the 286 dealt hands.

        `current` is each hypothesised dealt hand shifted by the opponent's public net
        trades, (286, 4). `open_sells` counts their resting offers per suit, which tie up
        cards: an offer is only possible in suits where they have a free card.
        """
        score = self.scores(current)
        if side is Side.BUY:
            beta = self.beta * min(max(price / self.price_ref, 0.1), 2.0)
            logits = beta * score
            allowed = np.ones(score.shape, dtype=bool)
        else:
            beta = self.beta * min(max(self.price_ref / max(price, 1), 0.1), 2.0)
            logits = -beta * score
            allowed = current - open_sells >= 1
        p = normalise_log(np.where(allowed, logits, NEG_INF), axis=1)[:, suit]
        n_allowed = np.maximum(allowed.sum(axis=1), 1)
        p = (1.0 - self.epsilon) * p + self.epsilon * allowed[:, suit] / n_allowed
        with np.errstate(divide="ignore"):
            return np.log(p)


# ---------------------------------------------------------------------------
# Running belief for one seat


class BeliefState:
    """Posterior over configurations for the player in `seat`, updated from the tape.

    With `model=None` this is hand-only inference: order flow is ignored entirely,
    including the hard selling constraints. Opponents' goal-suit holdings are still
    tracked from public trades, since that is bookkeeping rather than inference.

    `exact_joint=False` treats opponents' hands as independent given the configuration
    (mean-field). Kept only to measure what the exact joint buys.
    """

    def __init__(
        self,
        seat: int,
        initial_hand: Sequence[int],
        model: Level0Model | None = None,
        exact_joint: bool = True,
    ):
        self.seat = seat
        self.model = model
        self.initial_hand = np.asarray(initial_hand, dtype=np.int64)
        self.log_hand = hand_log_likelihood(self.initial_hand)
        self.opponents = [p for p in range(N_PLAYERS) if p != seat]
        self._row = {p: i for i, p in enumerate(self.opponents)}
        self.opp_log_prior = opponent_hand_log_prior(self.initial_hand)  # (12, 286)
        # An observer outside the game (seat -1, used for model fitting) has no joint to sum over.
        self.exact_joint = exact_joint and model is not None and len(self.opponents) == 3
        if self.exact_joint:
            self._rest_index, self._log_rest_fact = rest_index(self.initial_hand)
        self.behaviour = np.zeros((len(self.opponents), N_HANDS))
        self.flow = np.zeros((N_PLAYERS, N_SUITS), dtype=np.int64)  # net cards bought
        self.open_sells = np.zeros((N_PLAYERS, N_SUITS), dtype=np.int64)
        self.min_dealt = np.zeros((N_PLAYERS, N_SUITS), dtype=np.int64)  # lower bound on dealt counts
        self._resting: dict[int, tuple[int, int, Side]] = {}
        self.cursor = 0
        self.n_trades = 0
        self._evidence_changed = True  # opponents' hand likelihoods moved
        self._cache: dict[str, np.ndarray] = {}

    # -- evidence --------------------------------------------------------------

    def update(self, tape: Sequence[Event]) -> bool:
        n = len(tape)
        if n == self.cursor:
            return False
        for event in tape[self.cursor : n]:
            self._process(event)
        self.cursor = n
        self._cache.pop("q", None)
        if self._evidence_changed:
            self._cache.clear()
        self._evidence_changed = False
        return True

    def _process(self, ev: Event) -> None:
        if isinstance(ev, OrderPlaced):
            p = ev.player
            if self.model is not None and p != self.seat:
                current = HANDS + self.flow[p]
                self.behaviour[self._row[p]] += self.model.log_likelihood(
                    ev.side, ev.suit, ev.price, current, self.open_sells[p]
                )
                self._evidence_changed = True
            if not ev.is_take:
                self._resting[ev.order_id] = (p, ev.suit, ev.side)
                if ev.side is Side.SELL:
                    self.open_sells[p, ev.suit] += 1
            self._tighten(p, ev.suit, extra=1 if ev.is_take and ev.side is Side.SELL else 0)
        elif isinstance(ev, OrderCancelled):
            self._drop(ev.order_id)
        elif isinstance(ev, TradeEvent):
            self.flow[ev.buyer, ev.suit] += 1
            self.flow[ev.seller, ev.suit] -= 1
            self._drop(ev.maker_order_id)
            self._drop(ev.taker_order_id)
            self._tighten(ev.seller, ev.suit)
            self.n_trades += 1

    def _drop(self, order_id: int) -> None:
        info = self._resting.pop(order_id, None)
        if info is not None and info[2] is Side.SELL:
            self.open_sells[info[0], info[1]] -= 1

    def _tighten(self, p: int, suit: int, extra: int = 0) -> None:
        # dealt + net bought >= cards currently tied up in offers (+1 for a sell about to fill)
        bound = self.open_sells[p, suit] + extra - self.flow[p, suit]
        if bound > self.min_dealt[p, suit]:
            self.min_dealt[p, suit] = bound
            if p != self.seat and self.model is not None:
                self._evidence_changed = True

    # -- queries ---------------------------------------------------------------

    def opponent_log_likelihood(self) -> np.ndarray:
        """log P(observed flow | opponent dealt hand), shape (3, 286)."""
        if self.model is None:
            return np.zeros((len(self.opponents), N_HANDS))
        bounds = self.min_dealt[self.opponents]  # (3, 4)
        feasible = (HANDS[None, :, :] >= bounds[:, None, :]).all(axis=-1)
        return np.where(feasible, self.behaviour, NEG_INF)

    def _hand_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """(log evidence per config (12,), P(opponent j dealt h | config, flow) (3, 12, 286))."""
        if "weights" not in self._cache:
            log_lik = self.opponent_log_likelihood()
            if self.model is not None and self.exact_joint:
                evidence, weights = joint_hand_posterior(log_lik, self._rest_index, self._log_rest_fact)
            else:
                # Mean-field. With no flow model this is exact: each opponent's marginal
                # hand really is hypergeometric and the evidence is 1 for every config.
                log_w = self.opp_log_prior[None, :, :] + log_lik[:, None, :]
                evidence = logsumexp(log_w, axis=-1).sum(axis=0)
                weights = normalise_log(log_w, axis=-1)
            self._cache["evidence"], self._cache["weights"] = evidence, weights
        return self._cache["evidence"], self._cache["weights"]

    def config_posterior(self) -> np.ndarray:
        if "post" not in self._cache:
            evidence, _ = self._hand_weights()
            post = normalise_log(self.log_hand + evidence)
            if post.sum() == 0:  # only reachable if the model assigns zero probability to the truth
                post = hand_posterior(self.initial_hand)
            self._cache["post"] = post
        return self._cache["post"]

    def goal_probs(self) -> np.ndarray:
        return goal_marginal(self.config_posterior())

    def opponent_goal_count_dists(self) -> np.ndarray:
        """P(opponent j currently holds x goal cards | config c), shape (12, 3, 11)."""
        if "q" not in self._cache:
            _, w = self._hand_weights()  # (3, 12, 286)
            dealt_goal = HANDS[:, CONFIG_GOAL].T  # (12, 286)
            bought_goal = self.flow[self.opponents][:, CONFIG_GOAL]  # (3, 12)
            held = np.clip(dealt_goal[None, :, :] + bought_goal[:, :, None], 0, MAX_GOAL_HELD)
            n_opp = len(self.opponents)
            index = (np.arange(N_CONFIGS)[None, :, None] * n_opp + np.arange(n_opp)[:, None, None]) * (
                MAX_GOAL_HELD + 1
            ) + held
            size = N_CONFIGS * n_opp * (MAX_GOAL_HELD + 1)
            q = np.bincount(index.ravel(), weights=w.ravel(), minlength=size)
            self._cache["q"] = q.reshape(N_CONFIGS, n_opp, MAX_GOAL_HELD + 1)
        return self._cache["q"]


# ---------------------------------------------------------------------------
# Reference implementation (slow): brute-force enumeration of the joint deal


def exact_config_posterior(my_hand: Sequence[int], opp_log_lik: np.ndarray) -> np.ndarray:
    """P(config | my hand, flow) by enumerating every (h_B, h_C, h_D) deal of the other 30 cards.

    `opp_log_lik` is (3, 286): log P(flow of opponent j | j's dealt hand). Used in tests
    to check `joint_hand_posterior`.
    """
    my_hand = np.asarray(my_hand, dtype=np.int64)
    log_post = hand_log_likelihood(my_hand)
    rest = DECK_SIZE - HAND_SIZE
    norm = log_comb(rest, HAND_SIZE) + log_comb(rest - HAND_SIZE, HAND_SIZE)
    index = {tuple(h): i for i, h in enumerate(HANDS.tolist())}
    for c in range(N_CONFIGS):
        if not np.isfinite(log_post[c]):
            continue
        remaining = CONFIG_COUNTS[c] - my_hand
        hd = remaining - HANDS[:, None, :] - HANDS[None, :, :]  # (286, 286, 4)
        ib, ic = np.nonzero((hd >= 0).all(axis=-1))
        id_ = np.array([index[tuple(row)] for row in hd[ib, ic].tolist()], dtype=np.int64)
        log_deal = (log_comb(remaining, HANDS[ib]) + log_comb(remaining - HANDS[ib], HANDS[ic])).sum(axis=-1) - norm
        total = log_deal + opp_log_lik[0, ib] + opp_log_lik[1, ic] + opp_log_lik[2, id_]
        log_post[c] += logsumexp(total) if total.size else NEG_INF
    return normalise_log(log_post)
