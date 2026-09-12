from __future__ import annotations

import numpy as np

from ..inference import BeliefState, Level0Model
from ..market import Side
from ..protocol import Action, Observation, OrderPlaced, Pass, PlaceOrder, Take
from ..valuation import card_values
from .trader import ValuationTrader


class BayesianTrader(ValuationTrader):
    """Exact posterior over the 12 configurations, inventory-aware card values.

    Ablations: use_order_flow=False is hand-only inference. exact_joint=False treats
    opponents' hands as independent (mean-field). inventory_aware=False values every
    goal card at the flat average pot/n instead of its marginal value.
    """

    name = "bayes"

    def __init__(
        self,
        seat: int,
        seed: int,
        use_order_flow: bool = True,
        opponent_model: Level0Model | None = None,
        inventory_aware: bool = True,
        exact_joint: bool = True,
        **kwargs,
    ):
        super().__init__(seat, seed, **kwargs)
        self.use_order_flow = use_order_flow
        self.opponent_model = opponent_model or Level0Model()
        self.inventory_aware = inventory_aware
        self.exact_joint = exact_joint
        self.belief: BeliefState | None = None
        self._values: tuple[np.ndarray, np.ndarray] | None = None
        self._values_hand: tuple[int, ...] | None = None
        self._beliefs_by_trade: dict[int, np.ndarray] = {}

    def valuations(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        if self.belief is None:
            model = self.opponent_model if self.use_order_flow else None
            self.belief = BeliefState(self.seat, obs.initial_hand, model, self.exact_joint)
            self._beliefs_by_trade[0] = self.belief.goal_probs()
        changed = self.belief.update(obs.tape)
        if changed or self._values is None or obs.hand != self._values_hand:
            posterior = self.belief.config_posterior()
            q = self.belief.opponent_goal_count_dists()
            self._values = card_values(posterior, q, obs.hand, self.inventory_aware)
            self._values_hand = obs.hand
            if changed:
                self._beliefs_by_trade[self.belief.n_trades] = self.belief.goal_probs()
        return self._values

    def diagnostics(self) -> dict:
        trades = sorted(self._beliefs_by_trade)
        return {
            "belief_trades": np.array(trades, dtype=np.int32),
            "goal_probs": np.array([self._beliefs_by_trade[t] for t in trades], dtype=np.float32),
        }


class ConcealingBayesianTrader(BayesianTrader):
    """A Bayesian trader that tries to leak less about what it believes.

    cooldown: after a buy-side action in a suit it favours (P(goal) >= favourite_threshold),
        it waits this many turns before the next one, so its interest is spread out.
    decoy_rate: per turn, the chance of bidding `decoy_price` in a suit it thinks is
        unlikely to be the goal. The decoy is left resting for `decoy_hold` turns before
        the usual stale-quote logic may cancel it, so it can genuinely get filled.
    """

    name = "bayes_conceal"

    def __init__(
        self,
        seat: int,
        seed: int,
        cooldown: int = 0,
        favourite_threshold: float = 0.4,
        decoy_rate: float = 0.0,
        decoy_price: int = 6,
        decoy_hold: int = 6,
        decoy_max_prob: float = 0.15,
        **kwargs,
    ):
        super().__init__(seat, seed, **kwargs)
        self.cooldown = cooldown
        self.favourite_threshold = favourite_threshold
        self.decoy_rate = decoy_rate
        self.decoy_price = decoy_price
        self.decoy_hold = decoy_hold
        self.decoy_max_prob = decoy_max_prob
        self._last_favourite_buy = -(10**9)
        self._decoy_pending = False
        self._tape_mark = 0
        self.n_decoys = 0

    def act(self, obs: Observation) -> Action:
        if self._decoy_pending:
            for ev in reversed(obs.tape[self._tape_mark :]):
                if isinstance(ev, OrderPlaced) and ev.player == self.seat:
                    self.protected[ev.order_id] = obs.turn + self.decoy_hold
                    break
            self._decoy_pending = False
        self._tape_mark = len(obs.tape)

        self.valuations(obs)  # builds the belief on the first turn and refreshes card values
        goal = self.belief.goal_probs()

        if self.decoy_rate and self.rng.random() < self.decoy_rate:
            decoy = self._decoy(obs, goal)
            if decoy is not None:
                self._decoy_pending = True
                self.n_decoys += 1
                return decoy

        action = super().act(obs)
        if self.cooldown and self._is_favourite_buy(action, goal):
            if obs.turn - self._last_favourite_buy < self.cooldown:
                return Pass()
            self._last_favourite_buy = obs.turn
        return action

    def _is_favourite_buy(self, action: Action, goal: np.ndarray) -> bool:
        if isinstance(action, (PlaceOrder, Take)) and action.side is Side.BUY:
            return goal[action.suit] >= self.favourite_threshold
        return False

    def _decoy(self, obs: Observation, goal: np.ndarray) -> Action | None:
        price = self.decoy_price
        if obs.free_chips() < price:
            return None
        options = []
        for s in np.flatnonzero(goal < self.decoy_max_prob):
            s = int(s)
            book = obs.books[s]
            ask = book.best_ask(exclude_player=self.seat)
            bid = book.best_bid(exclude_player=self.seat)
            if obs.my_orders(s, Side.BUY) or (ask is not None and ask.price <= price):
                continue
            if bid is None or bid.price < price:
                options.append(s)
        return PlaceOrder(self.rng.choice(options), Side.BUY, price) if options else None

    def diagnostics(self) -> dict:
        return {**super().diagnostics(), "n_decoys": self.n_decoys}
