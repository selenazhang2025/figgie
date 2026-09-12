import random
from collections import Counter

from figgie.deck import HANDS, N_HANDS, PARTNER, all_configs, deal, sample_config


def test_twelve_configurations():
    configs = all_configs()
    assert len(configs) == 12
    assert len(set(configs)) == 12
    for c in configs:
        assert sum(c.counts) == 40
        assert sorted(c.counts) in ([8, 10, 10, 12],)
        assert c.goal_suit == PARTNER[c.long_suit]
        assert c.goal_count in (8, 10)
        assert c.bonus == (120 if c.goal_count == 8 else 100)


def test_three_configs_per_long_suit_and_goal_split():
    configs = all_configs()
    by_long = Counter(c.long_suit for c in configs)
    assert set(by_long.values()) == {3}
    assert Counter(c.goal_count for c in configs) == {8: 4, 10: 8}


def test_hand_enumeration():
    assert N_HANDS == 286
    assert (HANDS.sum(axis=1) == 10).all()
    assert len({tuple(h) for h in HANDS}) == 286


def test_deal_uses_exact_deck():
    rng = random.Random(0)
    for _ in range(50):
        config = sample_config(rng)
        hands = deal(config, rng)
        assert all(sum(h) == 10 for h in hands)
        assert [sum(h[s] for h in hands) for s in range(4)] == list(config.counts)
