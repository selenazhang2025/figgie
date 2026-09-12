from math import comb

import numpy as np
import pytest

from figgie.deck import CONFIG_GOAL, CONFIG_GOAL_COUNT, POT, all_configs
from figgie.valuation import PRIOR_Q, card_values, expected_shares, marginal_value_curve

PRIOR = np.broadcast_to(PRIOR_Q, (3, len(PRIOR_Q)))


def shares(n_goal, k):
    return expected_shares(PRIOR, np.array(n_goal), np.array(k))


@pytest.mark.parametrize("n_goal", [8, 10])
def test_expected_bonus_share_is_a_quarter_by_symmetry(n_goal):
    # Before any trading every seat is exchangeable, so each expects a quarter of the bonus.
    total = 0.0
    for k in range(n_goal + 1):
        p_k = comb(n_goal, k) * comb(40 - n_goal, 10 - k) / comb(40, 10)
        total += p_k * shares(n_goal, k)[0]
    assert total == pytest.approx(0.25, abs=1e-12)


@pytest.mark.parametrize("n_goal", [8, 10])
def test_buying_from_an_opponent_matches_reconditioning_under_the_deal_prior(n_goal):
    for k in range(n_goal):
        assert shares(n_goal, k)[1] == pytest.approx(shares(n_goal, k + 1)[0], abs=1e-12)


@pytest.mark.parametrize("n_goal", [8, 10])
def test_marginal_value_is_hump_shaped_and_bounded(n_goal):
    curve = marginal_value_curve(n_goal)
    mv = curve["marginal_value"]
    bonus = POT - 10 * n_goal
    assert (mv >= 10 - 1e-9).all() and (mv <= 10 + bonus + 1e-9).all()
    peak = int(mv.argmax())
    assert 0 < peak < len(mv) - 1
    assert (np.diff(mv[: peak + 1]) >= -1e-9).all()
    assert (np.diff(mv[peak:]) <= 1e-9).all()
    assert mv[0] < mv[peak] / 2 and mv[-1] < mv[peak] / 2
    assert (np.diff(curve["payout"]) > 0).all()


def test_shares_are_monotone_in_holdings():
    for n_goal in (8, 10):
        now = [shares(n_goal, k)[0] for k in range(n_goal + 1)]
        assert (np.diff(now) >= -1e-12).all()
        assert now[0] == 0 and now[-1] == 1


def test_card_values_with_known_config_match_the_curve():
    for c, config in enumerate(all_configs()):
        probs = np.eye(12)[c]
        q = np.broadcast_to(PRIOR_Q, (12, 3, len(PRIOR_Q)))
        curve = marginal_value_curve(config.goal_count)
        for k in range(config.goal_count):
            hand = [0, 0, 0, 0]
            hand[config.goal_suit] = k
            buy, sell = card_values(probs, q, hand)
            assert buy[config.goal_suit] == pytest.approx(curve["marginal_value"][k])
            assert np.delete(buy, config.goal_suit).sum() == 0
            assert (sell[config.goal_suit] == 0) == (k == 0)


def test_bid_ceiling_scales_with_goal_probability():
    q = np.broadcast_to(PRIOR_Q, (12, 3, len(PRIOR_Q)))
    probs = np.full(12, 1 / 12)
    hand = (2, 2, 3, 3)
    buy, _ = card_values(probs, q, hand)
    for s in range(4):
        per_config = [card_values(np.eye(12)[c], q, hand)[0][s] for c in np.flatnonzero(CONFIG_GOAL == s)]
        assert buy[s] == pytest.approx(sum(per_config) / 12)


def test_flat_valuation_ignores_inventory():
    q = np.broadcast_to(PRIOR_Q, (12, 3, len(PRIOR_Q)))
    for c in range(12):
        buy, _ = card_values(np.eye(12)[c], q, (3, 3, 2, 2), inventory_aware=False)
        assert buy[CONFIG_GOAL[c]] == pytest.approx(POT / CONFIG_GOAL_COUNT[c])
