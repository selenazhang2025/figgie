import pytest

from figgie.agents import (
    AgentSpec,
    BayesianTrader,
    ConcealingBayesianTrader,
    LongSuitHeuristic,
    RandomAgent,
    ScarcityHeuristic,
)
from figgie.deck import CLUBS, DIAMONDS, POT, STARTING_CHIPS
from figgie.engine import Game
from figgie.tournament import FocalRecord, focal_table, paired_diff_ci, play, run_focal

TABLE = [
    AgentSpec(BayesianTrader),
    AgentSpec(ConcealingBayesianTrader, {"cooldown": 3, "decoy_rate": 0.3}),
    AgentSpec(LongSuitHeuristic),
    AgentSpec(ScarcityHeuristic),
]


@pytest.mark.parametrize("seed", range(4))
def test_agents_only_take_legal_actions(seed):
    agents = [spec.build(seat, seed * 4 + seat) for seat, spec in enumerate(TABLE)]
    game = Game(agents, seed=seed, n_turns=60, strict=True)
    result = game.run(on_step=lambda g: g.check_invariants())
    assert sum(result.payouts) == POT
    assert sum(result.final_chips) == 4 * STARTING_CHIPS


def test_games_are_reproducible():
    assert play(11, TABLE, n_turns=40).final_chips == play(11, TABLE, n_turns=40).final_chips


def test_heuristic_targets():
    scarcity = ScarcityHeuristic(0, 0)
    long_suit = LongSuitHeuristic(0, 0)
    hand = (5, 2, 2, 1)
    assert scarcity.target_suit(hand) == DIAMONDS
    assert long_suit.target_suit(hand) == CLUBS


def test_focal_table_rotates_seat_and_preserves_field_order():
    field = [AgentSpec(RandomAgent, label=f"f{i}") for i in range(3)]
    focal = AgentSpec(BayesianTrader, label="focal")
    for seed in range(8):
        specs, seat = focal_table(focal, field, seed)
        assert seat == seed % 4 and specs[seat] is focal
        assert [s for i, s in enumerate(specs) if i != seat] == field


def test_paired_conditions_share_deals():
    field = [AgentSpec(LongSuitHeuristic), AgentSpec(ScarcityHeuristic), AgentSpec(RandomAgent)]
    conditions = {"a": AgentSpec(BayesianTrader), "b": AgentSpec(RandomAgent)}
    out = run_focal(conditions, field, seeds=range(6), n_turns=30, workers=1)
    for ra, rb in zip(out["a"], out["b"]):
        assert (ra.seed, ra.seat, ra.goal_suit, ra.goal_count) == (rb.seed, rb.seat, rb.goal_suit, rb.goal_count)
    estimate = paired_diff_ci(out["a"], out["b"])
    assert estimate.n == 6


def test_paired_diff_ci():
    def rec(seed, profit):
        return FocalRecord("x", seed, 0, profit, 0, 8, 0, 0, [])

    est = paired_diff_ci([rec(0, 10), rec(1, 20)], [rec(1, 5), rec(0, 4)])
    assert est.mean == pytest.approx(10.5)
