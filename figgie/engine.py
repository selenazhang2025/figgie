"""Game engine: deal, run the market in discrete turns, settle.

The engine owns all hidden state (deck configuration, every hand, every chip
count). Agents are handed an `Observation` built fresh each time they act and
return one `Action`. Nothing else crosses the boundary.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from numbers import Integral

from .deck import (
    ANTE,
    CARD_PAYOUT,
    N_PLAYERS,
    N_SUITS,
    POT,
    STARTING_CHIPS,
    DeckConfig,
    deal,
    sample_config,
)
from .market import Fill, Order, OrderBook, Side
from .protocol import (
    Action,
    BookView,
    CancelOrder,
    Event,
    Observation,
    OrderCancelled,
    OrderPlaced,
    Pass,
    PlaceOrder,
    Quote,
    Take,
    TapeView,
    TradeEvent,
)

# Four minutes of trading discretised into turns; each turn every player acts once
# in a freshly shuffled order.
DEFAULT_TURNS = 120
MAX_PRICE = 200


class InvalidAction(Exception):
    pass


@dataclass
class GameResult:
    seed: int
    config: DeckConfig
    agent_names: list[str]
    initial_hands: list[list[int]]
    final_hands: list[list[int]]
    payouts: list[int]
    final_chips: list[int]
    n_trades: int
    n_events: int
    invalid_actions: list[int]
    diagnostics: list[dict] = field(default_factory=list)

    @property
    def profits(self) -> list[int]:
        return [c - STARTING_CHIPS for c in self.final_chips]

    @property
    def goal_counts(self) -> list[int]:
        return [h[self.config.goal_suit] for h in self.final_hands]


def settle(goal_counts: Sequence[int], pot: int = POT) -> list[int]:
    """Pay $10 per goal card, split the remainder among players holding the most.

    Chips are integers, so an indivisible remainder goes one chip at a time to the
    tied winners in seat order. The payouts always sum to exactly the pot.
    """
    payouts = [CARD_PAYOUT * g for g in goal_counts]
    bonus = pot - sum(payouts)
    if bonus < 0:
        raise ValueError("goal cards pay out more than the pot")
    best = max(goal_counts)
    winners = [p for p, g in enumerate(goal_counts) if g == best]
    share, remainder = divmod(bonus, len(winners))
    for i, p in enumerate(winners):
        payouts[p] += share + (1 if i < remainder else 0)
    return payouts


class Game:
    def __init__(
        self,
        agents: Sequence,
        seed: int,
        n_turns: int = DEFAULT_TURNS,
        config: DeckConfig | None = None,
        hands: list[list[int]] | None = None,
        strict: bool = False,
    ):
        if len(agents) != N_PLAYERS:
            raise ValueError(f"need {N_PLAYERS} agents")
        deal_rng = random.Random(f"figgie-deal-{seed}")
        self.seed = seed
        self.n_turns = n_turns
        self.strict = strict
        self._agents = list(agents)
        self._config = config if config is not None else sample_config(deal_rng)
        dealt = hands if hands is not None else deal(self._config, deal_rng)
        self._initial_hands = [list(h) for h in dealt]
        self._hands = [list(h) for h in dealt]
        self._chips = [STARTING_CHIPS - ANTE] * N_PLAYERS
        self._pot = POT
        self._committed_chips = [0] * N_PLAYERS
        self._committed_cards = [[0] * N_SUITS for _ in range(N_PLAYERS)]
        self._books = [OrderBook(s) for s in range(N_SUITS)]
        self._order_suit: dict[int, int] = {}
        self._tape: list[Event] = []
        self._tape_view = TapeView(self._tape)
        self._order_rng = random.Random(f"figgie-order-{seed}")
        self._next_order_id = 0
        self._next_seq = 0
        self._n_trades = 0
        self._invalid = [0] * N_PLAYERS
        self._book_cache: tuple[BookView, ...] | None = None
        self._turn = 0

    @property
    def tape(self) -> TapeView:
        """The public event log. For post-game analysis; agents get it via observations."""
        return self._tape_view

    # -- running ---------------------------------------------------------------

    def run(self, on_step: Callable[[Game], None] | None = None) -> GameResult:
        for turn in range(self.n_turns):
            self._turn = turn
            order = list(range(N_PLAYERS))
            self._order_rng.shuffle(order)
            for p in order:
                action = self._agents[p].act(self._observe(p))
                self.apply(p, action)
                if on_step is not None:
                    on_step(self)
        return self._finish()

    def _finish(self) -> GameResult:
        goal = self._config.goal_suit
        payouts = settle([h[goal] for h in self._hands], self._pot)
        final_chips = [c + pay for c, pay in zip(self._chips, payouts)]
        return GameResult(
            seed=self.seed,
            config=self._config,
            agent_names=[getattr(a, "name", type(a).__name__) for a in self._agents],
            initial_hands=[list(h) for h in self._initial_hands],
            final_hands=[list(h) for h in self._hands],
            payouts=payouts,
            final_chips=final_chips,
            n_trades=self._n_trades,
            n_events=len(self._tape),
            invalid_actions=list(self._invalid),
            diagnostics=[a.diagnostics() if hasattr(a, "diagnostics") else {} for a in self._agents],
        )

    def _observe(self, p: int) -> Observation:
        if self._book_cache is None:
            self._book_cache = tuple(
                BookView(
                    bids=tuple(Quote(o.order_id, o.player, o.price) for o in b.bids),
                    asks=tuple(Quote(o.order_id, o.player, o.price) for o in b.asks),
                )
                for b in self._books
            )
        return Observation(
            player=p,
            turn=self._turn,
            n_turns=self.n_turns,
            initial_hand=tuple(self._initial_hands[p]),
            hand=tuple(self._hands[p]),
            chips=self._chips[p],
            books=self._book_cache,
            tape=self._tape_view,
            hand_sizes=tuple(sum(h) for h in self._hands),
        )

    # -- actions ---------------------------------------------------------------

    def apply(self, p: int, action: Action) -> None:
        match action:
            case Pass():
                return
            case PlaceOrder(suit=suit, side=side, price=price):
                if self._valid_order(p, suit, side, price):
                    self._submit(p, suit, Side(side), price, is_take=False)
                    return
            case Take(suit=suit, side=side):
                if self._valid_suit(suit) and side in (Side.BUY, Side.SELL):
                    contra = self._books[suit].best(Side(side).opposite, exclude_player=p)
                    if contra is not None and self._valid_order(p, suit, side, contra.price):
                        self._submit(p, suit, Side(side), contra.price, is_take=True)
                        return
            case CancelOrder(order_id=order_id):
                suit = self._order_suit.get(order_id)
                if suit is not None:
                    order = self._books[suit].get(order_id)
                    if order is not None and order.player == p:
                        self._books[suit].cancel(order_id)
                        self._on_removed(order, "player")
                        return
        self._invalid[p] += 1
        if self.strict:
            raise InvalidAction(f"player {p}: {action!r}")

    def _valid_suit(self, suit) -> bool:
        return isinstance(suit, Integral) and 0 <= suit < N_SUITS

    def _valid_order(self, p: int, suit, side, price) -> bool:
        if not self._valid_suit(suit) or side not in (Side.BUY, Side.SELL):
            return False
        if not isinstance(price, Integral) or not 1 <= price <= MAX_PRICE:
            return False
        if side == Side.BUY:
            return self._chips[p] - self._committed_chips[p] >= price
        return self._hands[p][suit] - self._committed_cards[p][suit] >= 1

    def _submit(self, p: int, suit: int, side: Side, price: int, is_take: bool) -> None:
        suit, price = int(suit), int(price)
        order = Order(self._next_order_id, p, suit, side, price, self._next_seq)
        self._next_order_id += 1
        self._emit(OrderPlaced(self._next_seq, self._turn, order.order_id, p, suit, side, price, is_take))
        fill, stp_cancelled = self._books[suit].submit(order, ioc=is_take)
        for cancelled in stp_cancelled:
            self._on_removed(cancelled, "self_trade")
        if fill is not None:
            self._on_fill(fill)
        elif not is_take:
            self._order_suit[order.order_id] = suit
            if side is Side.BUY:
                self._committed_chips[p] += price
            else:
                self._committed_cards[p][suit] += 1
        self._book_cache = None

    def _on_removed(self, order: Order, reason: str) -> None:
        self._release(order)
        self._emit(
            OrderCancelled(
                self._next_seq, self._turn, order.order_id, order.player, order.suit, order.side, order.price, reason
            )
        )
        self._book_cache = None

    def _release(self, order: Order) -> None:
        del self._order_suit[order.order_id]
        if order.side is Side.BUY:
            self._committed_chips[order.player] -= order.price
        else:
            self._committed_cards[order.player][order.suit] -= 1

    def _on_fill(self, fill: Fill) -> None:
        self._release(fill.maker)
        suit, price, buyer, seller = fill.maker.suit, fill.price, fill.buyer, fill.seller
        self._chips[buyer] -= price
        self._chips[seller] += price
        self._hands[buyer][suit] += 1
        self._hands[seller][suit] -= 1
        self._n_trades += 1
        self._emit(
            TradeEvent(
                self._next_seq,
                self._turn,
                suit,
                price,
                buyer,
                seller,
                fill.taker.player,
                fill.maker.order_id,
                fill.taker.order_id,
            )
        )

    def _emit(self, event: Event) -> None:
        self._tape.append(event)
        self._next_seq += 1

    # -- invariants (used by tests) ------------------------------------------------

    def check_invariants(self) -> None:
        for s in range(N_SUITS):
            assert sum(h[s] for h in self._hands) == self._config.counts[s], "cards not conserved"
        assert sum(self._chips) + self._pot == N_PLAYERS * STARTING_CHIPS, "chips not conserved"
        for p in range(N_PLAYERS):
            assert self._chips[p] >= 0
            assert 0 <= self._committed_chips[p] <= self._chips[p]
            for s in range(N_SUITS):
                assert 0 <= self._committed_cards[p][s] <= self._hands[p][s]
        resting_chips = [0] * N_PLAYERS
        resting_cards = [[0] * N_SUITS for _ in range(N_PLAYERS)]
        for book in self._books:
            for o in book.bids:
                resting_chips[o.player] += o.price
            for o in book.asks:
                resting_cards[o.player][o.suit] += 1
            if book.bids and book.asks:
                assert book.bids[0].price < book.asks[0].price, "crossed book"
        assert resting_chips == self._committed_chips
        assert resting_cards == self._committed_cards
