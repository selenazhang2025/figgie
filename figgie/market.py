"""A single-suit continuous double auction with price-time priority.

The book knows nothing about hands or chips: funding checks and settlement live
in the engine. Every order is for exactly one card. Trades print at the resting
(maker) order's price. Self-trade prevention cancels the resting order.
"""

from __future__ import annotations

from bisect import insort
from dataclasses import dataclass
from enum import IntEnum


class Side(IntEnum):
    BUY = 0
    SELL = 1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


@dataclass(frozen=True, slots=True)
class Order:
    order_id: int
    player: int
    suit: int
    side: Side
    price: int
    seq: int  # arrival sequence number, the "time" in price-time priority


@dataclass(frozen=True, slots=True)
class Fill:
    maker: Order
    taker: Order

    @property
    def price(self) -> int:
        return self.maker.price

    @property
    def buyer(self) -> int:
        return self.maker.player if self.maker.side is Side.BUY else self.taker.player

    @property
    def seller(self) -> int:
        return self.maker.player if self.maker.side is Side.SELL else self.taker.player


def _bid_key(o: Order) -> tuple[int, int]:
    return (-o.price, o.seq)


def _ask_key(o: Order) -> tuple[int, int]:
    return (o.price, o.seq)


class OrderBook:
    def __init__(self, suit: int):
        self.suit = suit
        self.bids: list[Order] = []  # best first: highest price, then earliest
        self.asks: list[Order] = []  # best first: lowest price, then earliest
        self._by_id: dict[int, Order] = {}

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, order_id: int) -> bool:
        return order_id in self._by_id

    def get(self, order_id: int) -> Order | None:
        return self._by_id.get(order_id)

    def best(self, side: Side, exclude_player: int | None = None) -> Order | None:
        for o in self.bids if side is Side.BUY else self.asks:
            if o.player != exclude_player:
                return o
        return None

    def submit(self, order: Order, ioc: bool = False) -> tuple[Fill | None, list[Order]]:
        """Match an incoming order; rest it unless filled or immediate-or-cancel.

        Returns the fill (at most one, since orders are single-card) and any of the
        submitter's own resting orders cancelled by self-trade prevention.
        """
        assert order.suit == self.suit and order.price >= 1
        contra = self.asks if order.side is Side.BUY else self.bids
        stp_cancelled: list[Order] = []
        while contra:
            best = contra[0]
            crosses = best.price <= order.price if order.side is Side.BUY else best.price >= order.price
            if not crosses:
                break
            if best.player == order.player:
                stp_cancelled.append(self._remove(best))
                continue
            self._remove(best)
            return Fill(maker=best, taker=order), stp_cancelled
        if not ioc:
            self._rest(order)
        return None, stp_cancelled

    def cancel(self, order_id: int) -> Order | None:
        order = self._by_id.get(order_id)
        return self._remove(order) if order is not None else None

    def _rest(self, order: Order) -> None:
        if order.side is Side.BUY:
            insort(self.bids, order, key=_bid_key)
        else:
            insort(self.asks, order, key=_ask_key)
        self._by_id[order.order_id] = order

    def _remove(self, order: Order) -> Order:
        (self.bids if order.side is Side.BUY else self.asks).remove(order)
        del self._by_id[order.order_id]
        return order
