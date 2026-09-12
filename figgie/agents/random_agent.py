from __future__ import annotations

from ..deck import N_SUITS
from ..market import Side
from ..protocol import Action, CancelOrder, Observation, Pass, PlaceOrder, Take
from .base import Agent


class RandomAgent(Agent):
    """Uniformly random but always-legal play. The floor every other agent must clear."""

    name = "random"

    def __init__(self, seat: int, seed: int, pass_prob: float = 0.5, max_price: int = 20):
        super().__init__(seat, seed)
        self.pass_prob = pass_prob
        self.max_price = max_price

    def act(self, obs: Observation) -> Action:
        rng = self.rng
        if rng.random() < self.pass_prob:
            return Pass()
        roll = rng.random()
        suit = rng.randrange(N_SUITS)
        if roll < 0.4:
            price = rng.randint(1, self.max_price)
            if obs.free_chips() >= price:
                return PlaceOrder(suit, Side.BUY, price)
        elif roll < 0.8:
            held = [s for s in range(N_SUITS) if obs.free_cards(s) > 0]
            if held:
                return PlaceOrder(rng.choice(held), Side.SELL, rng.randint(1, self.max_price))
        elif roll < 0.95:
            side = rng.choice((Side.BUY, Side.SELL))
            book = obs.books[suit]
            if side is Side.BUY:
                ask = book.best_ask(exclude_player=self.seat)
                if ask is not None and obs.free_chips() >= ask.price:
                    return Take(suit, side)
            else:
                bid = book.best_bid(exclude_player=self.seat)
                if bid is not None and obs.free_cards(suit) > 0:
                    return Take(suit, side)
        else:
            mine = [q for b in obs.books for q in (*b.bids, *b.asks) if q.player == self.seat]
            if mine:
                return CancelOrder(rng.choice(mine).order_id)
        return Pass()
