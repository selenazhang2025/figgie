import dataclasses

import pytest

from figgie.agents import AgentSpec, RandomAgent
from figgie.deck import POT, STARTING_CHIPS, all_configs
from figgie.engine import Game, InvalidAction, settle
from figgie.market import Side
from figgie.protocol import (
    BookView,
    CancelOrder,
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


def random_agents(seed, **kwargs):
    return [AgentSpec(RandomAgent, kwargs).build(seat, seed * 10 + seat) for seat in range(4)]


class Scripted:
    """Plays a fixed list of actions, then passes."""

    def __init__(self, actions=()):
        self.actions = list(actions)

    def act(self, obs):
        return self.actions.pop(0) if self.actions else Pass()


@pytest.mark.parametrize("seed", range(20))
def test_cards_and_chips_conserved_after_every_action(seed):
    game = Game(random_agents(seed, pass_prob=0.1), seed=seed, n_turns=60, strict=True)
    result = game.run(on_step=lambda g: g.check_invariants())
    assert result.n_trades > 0
    assert sum(result.final_chips) == 4 * STARTING_CHIPS


@pytest.mark.parametrize("seed", range(20))
def test_settlement_pays_exactly_the_pot(seed):
    result = Game(random_agents(seed), seed=seed, n_turns=40).run()
    assert sum(result.payouts) == POT
    goal = result.config.goal_suit
    assert sum(h[goal] for h in result.final_hands) == result.config.goal_count


def test_settle_single_winner():
    assert settle([5, 2, 1, 0]) == [50 + 120, 20, 10, 0]
    assert settle([4, 3, 2, 1]) == [40 + 100, 30, 20, 10]


def test_settle_ties_split_and_sum_to_pot():
    # 10-card goal suit, bonus 100 split three ways: 34/33/33 with the odd chip in seat order.
    assert settle([3, 3, 3, 1]) == [64, 63, 63, 10]
    assert settle([4, 4, 1, 1]) == [90, 90, 10, 10]
    for goal_counts in ([2, 2, 2, 2], [0, 4, 4, 0], [3, 3, 1, 1]):
        assert sum(settle(goal_counts)) == POT


def test_same_seed_same_game():
    a = Game(random_agents(3), seed=3, n_turns=50).run()
    b = Game(random_agents(3), seed=3, n_turns=50).run()
    assert a.final_chips == b.final_chips and a.n_events == b.n_events


def test_deal_depends_only_on_seed():
    a = Game(random_agents(1), seed=42, n_turns=0).run()
    b = Game([Scripted() for _ in range(4)], seed=42, n_turns=0).run()
    assert a.config == b.config and a.initial_hands == b.initial_hands


def test_trade_moves_one_card_and_price_between_players():
    config = all_configs()[0]
    hands = [[3, 2, 3, 2], [3, 2, 2, 3], [3, 2, 3, 2], [3, 2, 2, 3]]
    seller = Scripted([PlaceOrder(0, Side.SELL, 7)])
    buyer = Scripted([Pass(), Take(0, Side.BUY)])
    game = Game([seller, buyer, Scripted(), Scripted()], seed=0, n_turns=2, config=config, hands=hands, strict=True)
    result = game.run(on_step=lambda g: g.check_invariants())
    assert result.n_trades == 1
    assert result.final_hands[0][0] == 2 and result.final_hands[1][0] == 4
    paid = [c - pay - (STARTING_CHIPS - 50) for c, pay in zip(result.final_chips, result.payouts)]
    assert paid[:2] == [7, -7]


def test_invalid_actions_rejected():
    config = all_configs()[0]
    hands = [[10, 0, 0, 0], [0, 8, 2, 0], [0, 0, 8, 2], [2, 0, 0, 8]]
    game = Game([Scripted() for _ in range(4)], seed=0, config=config, hands=hands, strict=True)
    with pytest.raises(InvalidAction):
        game.apply(0, PlaceOrder(1, Side.SELL, 5))  # no clubs to sell
    with pytest.raises(InvalidAction):
        game.apply(0, PlaceOrder(0, Side.BUY, 301))  # more than your chips
    with pytest.raises(InvalidAction):
        game.apply(0, Take(0, Side.BUY))  # nothing to take
    with pytest.raises(InvalidAction):
        game.apply(0, PlaceOrder(0, Side.BUY, 0))  # prices start at 1
    game.apply(1, PlaceOrder(1, Side.SELL, 5))
    with pytest.raises(InvalidAction):
        game.apply(0, CancelOrder(0))  # not your order
    with pytest.raises(InvalidAction):
        game.apply(1, Take(1, Side.BUY))  # can't take your own offer
    game.check_invariants()


def test_cards_committed_to_offers_cannot_be_sold_twice():
    config = all_configs()[0]
    hands = [[1, 2, 3, 4], [3, 2, 3, 2], [4, 2, 2, 2], [4, 2, 2, 2]]
    game = Game([Scripted() for _ in range(4)], seed=0, config=config, hands=hands, strict=True)
    game.apply(0, PlaceOrder(0, Side.SELL, 5))
    with pytest.raises(InvalidAction):
        game.apply(0, PlaceOrder(0, Side.SELL, 6))
    game.check_invariants()


ALLOWED_OBSERVATION_TYPES = (
    int,
    str,
    bool,
    type(None),
    tuple,
    Side,
    Observation,
    BookView,
    Quote,
    TapeView,
    OrderPlaced,
    OrderCancelled,
    TradeEvent,
)


def _walk(value, seen):
    if id(value) in seen:
        return
    seen.add(id(value))
    assert isinstance(value, ALLOWED_OBSERVATION_TYPES), f"observation leaks {type(value).__name__}"
    if isinstance(value, (tuple, TapeView)):
        for v in value:
            _walk(v, seen)
    elif dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            _walk(getattr(value, f.name), seen)


def test_observation_contains_only_public_information():
    captured = []

    class Spy(RandomAgent):
        def act(self, obs):
            captured.append(obs)
            return super().act(obs)

    agents = [Spy(seat, seat, pass_prob=0.2) for seat in range(4)]
    game = Game(agents, seed=7, n_turns=30)
    result = game.run()
    for obs in captured[::17]:
        _walk(obs, set())
        field_names = {f.name for f in dataclasses.fields(obs)}
        assert field_names == {
            "player", "turn", "n_turns", "initial_hand", "hand", "chips", "books", "tape", "hand_sizes",
        }
    # Each observation's hand is the observer's own, never anyone else's.
    first = {o.player: o for o in captured if o.turn == 0}
    assert {p: list(o.initial_hand) for p, o in first.items()} == dict(enumerate(result.initial_hands))


def test_tape_view_is_read_only():
    tape = TapeView([])
    assert not hasattr(tape, "append")
    with pytest.raises(TypeError):
        tape[0] = None  # type: ignore[index]


def test_a_trade_clears_every_resting_order():
    config = all_configs()[0]  # 12 spades, 8 clubs, 10 hearts, 10 diamonds
    hands = [[3, 2, 3, 2], [3, 2, 2, 3], [3, 2, 3, 2], [3, 2, 2, 3]]
    game = Game([Scripted() for _ in range(4)], seed=0, config=config, hands=hands, strict=True)
    game.apply(0, PlaceOrder(0, Side.SELL, 7))
    game.apply(1, PlaceOrder(1, Side.BUY, 5))
    game.apply(2, PlaceOrder(2, Side.SELL, 9))
    assert sum(len(b.bids) + len(b.asks) for b in game._books) == 3
    game.apply(3, Take(0, Side.BUY))  # one trade wipes the whole market
    assert sum(len(b.bids) + len(b.asks) for b in game._books) == 0
    cleared = [e for e in game.tape if isinstance(e, OrderCancelled) and e.reason == "trade_clear"]
    assert {(e.player, e.suit) for e in cleared} == {(1, 1), (2, 2)}
    game.check_invariants()


def test_books_survive_trades_when_clearing_is_disabled():
    config = all_configs()[0]
    hands = [[3, 2, 3, 2], [3, 2, 2, 3], [3, 2, 3, 2], [3, 2, 2, 3]]
    game = Game([Scripted() for _ in range(4)], seed=0, config=config, hands=hands, strict=True,
                clear_books_on_trade=False)
    game.apply(0, PlaceOrder(0, Side.SELL, 7))
    game.apply(1, PlaceOrder(1, Side.BUY, 5))
    game.apply(3, Take(0, Side.BUY))
    assert sum(len(b.bids) + len(b.asks) for b in game._books) == 1
    game.check_invariants()
