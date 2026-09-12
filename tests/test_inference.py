import random

import numpy as np
import pytest
from scipy.stats import multivariate_hypergeom

from figgie.agents import AgentSpec, RandomAgent
from figgie.deck import CLUBS, CONFIG_COUNTS, CONFIG_GOAL, HAND_INDEX, HANDS, SPADES, all_configs, deal
from figgie.engine import Game
from figgie.inference import (
    BeliefState,
    Level0Model,
    exact_config_posterior,
    goal_marginal,
    hand_log_likelihood,
    hand_posterior,
    log_comb,
    opponent_hand_log_prior,
)
from figgie.market import Side
from figgie.protocol import OrderPlaced, TradeEvent

CONFIGS = all_configs()


def test_hand_likelihood_matches_scipy_multivariate_hypergeometric():
    for c, counts in enumerate(CONFIG_COUNTS):
        for h in HANDS[::5]:
            expected = multivariate_hypergeom.pmf(x=h, m=counts, n=10)
            assert np.exp(hand_log_likelihood(h)[c]) == pytest.approx(expected, rel=1e-9, abs=1e-300)


def test_hand_likelihood_sums_to_one_over_all_hands():
    probs = np.exp(hand_log_likelihood(HANDS))  # (286, 12)
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, rtol=1e-12)


def test_posterior_sums_to_one_and_rules_out_impossible_configs():
    posts = hand_posterior(HANDS)
    np.testing.assert_allclose(posts.sum(axis=1), 1.0, rtol=1e-12)
    post = hand_posterior((0, 9, 1, 0))  # nine clubs: clubs cannot be the 8-card suit
    assert post[CONFIG_COUNTS[:, CLUBS] == 8].sum() == 0
    assert post[CONFIG_COUNTS[:, CLUBS] >= 10].sum() == pytest.approx(1.0)


def test_long_suit_points_at_its_partner():
    assert goal_marginal(hand_posterior((6, 1, 2, 1))).argmax() == CLUBS


def test_hand_posterior_is_calibrated():
    # Averaged over deals the posterior must return the prior, and the mean probability
    # placed on the true goal must equal the mean of sum_s P(s)^2.
    rng = random.Random(0)
    n = 20_000
    mean_post = np.zeros(12)
    on_truth = self_consistency = 0.0
    for _ in range(n):
        c = rng.randrange(12)
        post = hand_posterior(deal(CONFIGS[c], rng)[0])
        goal = goal_marginal(post)
        mean_post += post / n
        on_truth += goal[CONFIGS[c].goal_suit] / n
        self_consistency += (goal**2).sum() / n
    np.testing.assert_allclose(mean_post, 1 / 12, atol=0.006)
    assert on_truth == pytest.approx(self_consistency, abs=0.006)


def test_opponent_hand_prior_is_normalised():
    my_hand = (3, 3, 2, 2)
    lp = opponent_hand_log_prior(my_hand)
    feasible = hand_posterior(my_hand) > 0
    np.testing.assert_allclose(np.exp(lp[feasible]).sum(axis=1), 1.0, rtol=1e-12)


def synthetic_buys(model, config, rng, n_events, price=10):
    """Opponents 1-3 place buys drawn from the model itself, given their true hands."""
    hands = deal(config, rng)
    events = []
    zeros = np.zeros(4, dtype=np.int64)
    for t in range(n_events):
        p = 1 + t % 3
        idx = HAND_INDEX[tuple(hands[p])]
        probs = [np.exp(model.log_likelihood(Side.BUY, s, price, HANDS, zeros)[idx]) for s in range(4)]
        suit = rng.choices(range(4), weights=probs)[0]
        events.append(OrderPlaced(t, 0, t, p, suit, Side.BUY, price, is_take=True))
    return hands, events


def test_posterior_converges_to_truth_with_enough_synthetic_evidence():
    # Uses the fitted model the agents play with. (The "scarcity" model is not identifiable
    # enough for this: its bids reveal a player's shortest suit and little else.)
    model = Level0Model()
    rng = random.Random(1)
    for trial in range(8):
        config = CONFIGS[trial % 12]
        hands, events = synthetic_buys(model, config, rng, n_events=1500, price=20)
        belief = BeliefState(0, hands[0], model)
        belief.update(events)
        assert belief.config_posterior()[trial % 12] > 0.95


def test_joint_without_flow_is_the_hand_posterior():
    my_hand = (4, 2, 3, 1)
    brute = exact_config_posterior(my_hand, np.zeros((3, len(HANDS))))
    np.testing.assert_allclose(brute, hand_posterior(my_hand), atol=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_convolution_matches_brute_force_joint_enumeration(seed):
    model = Level0Model()
    rng = random.Random(seed)
    hands, events = synthetic_buys(model, CONFIGS[(seed * 5) % 12], rng, n_events=60)
    if hands[2][1] >= 2:  # add a hard constraint: seat 2 sold two of suit 1
        events += [TradeEvent(1000 + i, 0, 1, 3, 3, 2, 3, 5000 + i, 6000 + i) for i in range(2)]
    belief = BeliefState(0, hands[0], model)
    belief.update(events)
    brute = exact_config_posterior(hands[0], belief.opponent_log_likelihood())
    np.testing.assert_allclose(belief.config_posterior(), brute, atol=1e-9)


def test_joint_hand_marginals_match_brute_force():
    model = Level0Model()
    c = 4
    hands, events = synthetic_buys(model, CONFIGS[c], random.Random(7), n_events=45)
    belief = BeliefState(0, hands[0], model)
    belief.update(events)
    _, weights = belief._hand_weights()
    log_lik = belief.opponent_log_likelihood()

    remaining = CONFIG_COUNTS[c] - np.asarray(hands[0])
    marginal = np.zeros(len(HANDS))
    for ib, hb in enumerate(HANDS):
        hd = remaining - hb - HANDS  # every choice of C's hand at once
        ok = np.flatnonzero((hd >= 0).all(axis=1))
        id_ = [HAND_INDEX[tuple(int(x) for x in row)] for row in hd[ok]]
        log_deal = log_comb(remaining, hb).sum() + log_comb(remaining - hb, HANDS[ok]).sum(axis=1)
        marginal[ib] = np.exp(log_deal + log_lik[0, ib] + log_lik[1, ok] + log_lik[2, id_]).sum()
    np.testing.assert_allclose(weights[0, c], marginal / marginal.sum(), atol=1e-9)


def test_mean_field_is_measurably_worse_than_the_exact_joint():
    model = Level0Model()
    exact_on_truth, mean_field_on_truth = [], []
    for trial in range(3):
        config = CONFIGS[(trial * 5) % 12]
        hands, events = synthetic_buys(model, config, random.Random(100 + trial), n_events=1500, price=20)
        exact = BeliefState(0, hands[0], model)
        mean_field = BeliefState(0, hands[0], model, exact_joint=False)
        exact.update(events)
        mean_field.update(events)
        exact_on_truth.append(exact.goal_probs()[config.goal_suit])
        mean_field_on_truth.append(mean_field.goal_probs()[config.goal_suit])
    assert np.mean(exact_on_truth) > 0.95
    assert np.mean(exact_on_truth) > np.mean(mean_field_on_truth) + 0.2


def test_selling_constraints_are_exact_across_opponents():
    my_hand = (0, 4, 3, 3)
    # Seats 1 and 2 each sell me six spades. Either alone fits a 10-spade deck; together
    # they prove twelve spades exist, so spades is the long suit and clubs the goal.
    events = [TradeEvent(i, 0, SPADES, 1, 0, 1 + i % 2, 0, 1000 + i, 2000 + i) for i in range(12)]
    full = BeliefState(0, my_hand, Level0Model())
    full.update(events)
    assert full.goal_probs()[CLUBS] == pytest.approx(1.0)
    hand_only = BeliefState(0, my_hand, None)
    hand_only.update(events)
    assert hand_only.goal_probs()[CLUBS] < 0.9


@pytest.mark.parametrize("seed", range(10))
def test_bookkeeping_matches_engine_and_truth_is_never_ruled_out(seed):
    agents = [AgentSpec(RandomAgent, {"pass_prob": 0.1}).build(s, seed * 4 + s) for s in range(4)]
    game = Game(agents, seed=seed, n_turns=80)
    result = game.run()
    true_config = CONFIGS.index(result.config)
    net = np.array(result.final_hands) - np.array(result.initial_hands)
    for seat in range(4):
        belief = BeliefState(seat, result.initial_hands[seat], Level0Model())
        belief.update(game.tape)
        np.testing.assert_array_equal(belief.flow, net)
        np.testing.assert_array_equal(belief.open_sells, np.array(game._committed_cards))
        assert (belief.min_dealt <= np.array(result.initial_hands)).all()
        assert belief.config_posterior()[true_config] > 0
        assert belief.config_posterior().sum() == pytest.approx(1.0)


def test_opponent_goal_count_distributions_are_normalised_and_shifted_by_trades():
    belief = BeliefState(0, (3, 3, 2, 2), Level0Model())
    belief.update([TradeEvent(i, 0, CLUBS, 5, 1, 2, 1, 100 + i, 200 + i) for i in range(3)])
    q = belief.opponent_goal_count_dists()
    feasible = belief.config_posterior() > 0
    np.testing.assert_allclose(q[feasible].sum(axis=-1), 1.0)
    clubs_goal = feasible & (CONFIG_GOAL == CLUBS)
    x = np.arange(q.shape[-1])
    assert ((q[clubs_goal, 0] * x).sum(axis=-1) > 3).all()  # seat 1 bought three clubs
    assert belief.min_dealt[2, CLUBS] == 3  # seat 2 sold three, so was dealt at least three
