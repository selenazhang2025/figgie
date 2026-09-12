"""The narrow interface between engine and agents.

Everything here is immutable and contains only what a seated player could see:
their own hand and chips, the public books (quotes are attributed, as at a real
Figgie table), and the public tape of placements, cancels and trades. The deck
configuration and other players' hands never appear.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

from .market import Side

# ---------------------------------------------------------------------------
# Actions (agent -> engine)


@dataclass(frozen=True, slots=True)
class Pass:
    pass


@dataclass(frozen=True, slots=True)
class PlaceOrder:
    suit: int
    side: Side
    price: int


@dataclass(frozen=True, slots=True)
class CancelOrder:
    order_id: int


@dataclass(frozen=True, slots=True)
class Take:
    """Trade immediately against the best resting order of another player.

    `side` is the taker's side: BUY lifts the best offer, SELL hits the best bid.
    """

    suit: int
    side: Side


Action = Pass | PlaceOrder | CancelOrder | Take


# ---------------------------------------------------------------------------
# Public events (the tape)


@dataclass(frozen=True, slots=True)
class OrderPlaced:
    seq: int
    turn: int
    order_id: int
    player: int
    suit: int
    side: Side
    price: int
    is_take: bool  # True for Take actions (immediate-or-cancel at the touch)


@dataclass(frozen=True, slots=True)
class OrderCancelled:
    seq: int
    turn: int
    order_id: int
    player: int
    suit: int
    side: Side
    price: int
    reason: str  # "player" or "self_trade"


@dataclass(frozen=True, slots=True)
class TradeEvent:
    seq: int
    turn: int
    suit: int
    price: int
    buyer: int
    seller: int
    aggressor: int
    maker_order_id: int
    taker_order_id: int


Event = OrderPlaced | OrderCancelled | TradeEvent


class TapeView(Sequence):
    """Read-only window onto the engine's append-only event log."""

    __slots__ = ("_events",)

    def __init__(self, events: list[Event]):
        self._events = events

    def __len__(self) -> int:
        return len(self._events)

    @overload
    def __getitem__(self, i: int) -> Event: ...
    @overload
    def __getitem__(self, i: slice) -> tuple[Event, ...]: ...
    def __getitem__(self, i):
        if isinstance(i, slice):
            return tuple(self._events[i])
        return self._events[i]


# ---------------------------------------------------------------------------
# Observation (engine -> agent)


@dataclass(frozen=True, slots=True)
class Quote:
    order_id: int
    player: int
    price: int


@dataclass(frozen=True, slots=True)
class BookView:
    bids: tuple[Quote, ...]  # best first
    asks: tuple[Quote, ...]  # best first

    def best_bid(self, exclude_player: int | None = None) -> Quote | None:
        return next((q for q in self.bids if q.player != exclude_player), None)

    def best_ask(self, exclude_player: int | None = None) -> Quote | None:
        return next((q for q in self.asks if q.player != exclude_player), None)


@dataclass(frozen=True, slots=True)
class Observation:
    player: int
    turn: int
    n_turns: int
    initial_hand: tuple[int, ...]  # what you were dealt
    hand: tuple[int, ...]  # what you hold now, including cards committed to offers
    chips: int
    books: tuple[BookView, ...]  # one per suit
    tape: TapeView
    hand_sizes: tuple[int, ...]  # public: derivable from the tape

    def my_orders(self, suit: int, side: Side) -> tuple[Quote, ...]:
        book = self.books[suit]
        quotes = book.bids if side is Side.BUY else book.asks
        return tuple(q for q in quotes if q.player == self.player)

    def committed_chips(self) -> int:
        return sum(q.price for b in self.books for q in b.bids if q.player == self.player)

    def free_chips(self) -> int:
        return self.chips - self.committed_chips()

    def free_cards(self, suit: int) -> int:
        return self.hand[suit] - len(self.my_orders(suit, Side.SELL))
