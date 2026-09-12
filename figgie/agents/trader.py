from __future__ import annotations

import math

import numpy as np

from ..deck import N_SUITS
from ..market import Side
from ..protocol import Action, CancelOrder, Observation, Pass, PlaceOrder, Quote, Take
from .base import Agent


class ValuationTrader(Agent):
    """Turns per-suit card values into orders. Subclasses decide what cards are worth.

    Every non-random agent shares this execution logic, so comparisons between them
    isolate the valuation and inference, not the order handling. Each turn, in order:

    1. cancel a resting order that is now worse than its value (it would lose money if filled)
    2. take the resting order with the largest surplus over value, if above `take_edge`
    3. cancel a quote that has drifted more than `requote_tolerance` from its target
    4. post a new bid or offer that improves the touch, at value -/+ an edge
    """

    name = "trader"

    def __init__(
        self,
        seat: int,
        seed: int,
        take_edge: float = 1.0,
        quote_edge: float = 0.25,
        min_edge: float = 1.0,
        requote_tolerance: int = 2,
        min_ask: int = 2,
    ):
        super().__init__(seat, seed)
        self.take_edge = take_edge
        self.quote_edge = quote_edge
        self.min_edge = min_edge
        self.requote_tolerance = requote_tolerance
        self.min_ask = min_ask
        self.protected: dict[int, int] = {}  # order_id -> last turn it is exempt from cancels

    def valuations(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """(value of buying one more card, value lost by selling one) per suit."""
        raise NotImplementedError

    def act(self, obs: Observation) -> Action:
        buy_v, sell_v = self.valuations(obs)
        return (
            self._cancel_unprofitable(obs, buy_v, sell_v)
            or self._best_take(obs, buy_v, sell_v)
            or self._cancel_stale(obs, buy_v, sell_v)
            or self._best_quote(obs, buy_v, sell_v)
            or Pass()
        )

    def bid_target(self, value: float) -> int:
        return math.floor(value - max(self.min_edge, self.quote_edge * value))

    def ask_target(self, value: float) -> int:
        return max(self.min_ask, math.ceil(value + max(self.min_edge, self.quote_edge * value)))

    def _exempt(self, q: Quote, obs: Observation) -> bool:
        return self.protected.get(q.order_id, -1) >= obs.turn

    def _cancel_unprofitable(self, obs, buy_v, sell_v) -> Action | None:
        for s in range(N_SUITS):
            for q in obs.my_orders(s, Side.BUY):
                if q.price > buy_v[s] and not self._exempt(q, obs):
                    return CancelOrder(q.order_id)
            for q in obs.my_orders(s, Side.SELL):
                if q.price < sell_v[s] and not self._exempt(q, obs):
                    return CancelOrder(q.order_id)
        return None

    def _best_take(self, obs, buy_v, sell_v) -> Action | None:
        best, action = self.take_edge, None
        free_chips = obs.free_chips()
        for s in range(N_SUITS):
            book = obs.books[s]
            ask = book.best_ask(exclude_player=self.seat)
            if ask is not None and ask.price <= free_chips and buy_v[s] - ask.price >= best:
                best, action = buy_v[s] - ask.price, Take(s, Side.BUY)
            bid = book.best_bid(exclude_player=self.seat)
            if bid is not None and obs.hand[s] > 0 and bid.price - sell_v[s] >= best:
                best = bid.price - sell_v[s]
                # every card of this suit is tied up in my offers: free one up first
                action = Take(s, Side.SELL) if obs.free_cards(s) > 0 else CancelOrder(obs.my_orders(s, Side.SELL)[-1].order_id)
        return action

    def _cancel_stale(self, obs, buy_v, sell_v) -> Action | None:
        tol = self.requote_tolerance
        for s in range(N_SUITS):
            target = self.bid_target(buy_v[s])
            for i, q in enumerate(obs.my_orders(s, Side.BUY)):
                if (i > 0 or abs(q.price - target) > tol) and not self._exempt(q, obs):
                    return CancelOrder(q.order_id)
            target = self.ask_target(sell_v[s])
            for i, q in enumerate(obs.my_orders(s, Side.SELL)):
                if (i > 0 or abs(q.price - target) > tol) and not self._exempt(q, obs):
                    return CancelOrder(q.order_id)
        return None

    def _best_quote(self, obs, buy_v, sell_v) -> Action | None:
        candidates: list[Action] = []
        free_chips = obs.free_chips()
        for s in range(N_SUITS):
            book = obs.books[s]
            best_bid = book.best_bid(exclude_player=self.seat)
            best_ask = book.best_ask(exclude_player=self.seat)
            if not obs.my_orders(s, Side.BUY):
                price = self.bid_target(buy_v[s])
                if best_ask is not None:
                    price = min(price, best_ask.price - 1)
                if 1 <= price <= free_chips and (best_bid is None or price > best_bid.price):
                    candidates.append(PlaceOrder(s, Side.BUY, price))
            if not obs.my_orders(s, Side.SELL) and obs.free_cards(s) > 0:
                price = self.ask_target(sell_v[s])
                if best_bid is not None:
                    price = max(price, best_bid.price + 1)
                if best_ask is None or price < best_ask.price:
                    candidates.append(PlaceOrder(s, Side.SELL, price))
        return self.rng.choice(candidates) if candidates else None
