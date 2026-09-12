# Figgie: exact Bayesian inference over a hidden deck

Across 3,000 paired deals against tuned heuristic opponents, an agent that runs exact
Bayesian inference over the 12 possible decks, using its own hand and every opponent's
order flow, makes **+$45.9 per game** [+43.7, +48.0] out of a $350 stack.

Order flow is what earns it. The same agent reasoning from its hand alone loses $27.4 a
game by comparison, and names the goal suit in 38% of games against 71%.

The part I expected to matter most is the part that failed. A goal card's marginal value
under the majority bonus is hump-shaped in how many you already hold, and pricing off
that curve **costs $30.8 a game** against simply valuing every goal card at the flat
average. That result got stronger, not weaker, each time I made the simulation more
realistic or fixed a bug in my own code.

![Marginal value of a goal-suit card](figures/marginal_value.png)

The curve itself is real arithmetic. With an 8-card goal suit the third card is worth
$85, because it usually moves you into the lead for the $120 bonus, while the first is
worth $10 and the fifth $11. It is exactly right for one player trading against opponents
whose holdings stay put. [Section 3](#3-why-the-marginal-value-curve-loses-money) is about
what happens when everyone at the table prices off it.

---

## Results

All numbers are chips won per game. One focal agent plays a fixed field of three
opponents. Every row plays the **same 3,000 deals**: the seed fixes the deck, the hands,
the turn order and every seat's private randomness. Differences are paired game by game,
and brackets are 95% intervals.

### 1. Against tuned heuristics

The field is the long-suit heuristic, the scarcity heuristic and a hand-only Bayesian,
each with its own parameters tuned (see [benchmarks](#agents)).

| Focal agent | Chips per game | vs full Bayes, same deals | Goal suit correct |
|---|---|---|---|
| **Bayes, flat card value** | **+45.9** [+43.7, +48.0] | **+30.8** [+28.1, +33.5] | **78.2%** |
| Bayes, full (marginal value) | +15.1 [+12.5, +17.7] | – | 70.9% |
| Bayes, mean-field order flow | +2.5 [−0.0, +4.9] | −12.6 [−14.7, −10.5] | 64.5% |
| Scarcity heuristic | −1.0 [−2.2, +0.2] | −16.1 [−18.6, −13.5] | – |
| Long-suit heuristic | −1.9 [−4.2, +0.4] | −17.0 [−19.7, −14.2] | – |
| Bayes, hand only | −12.3 [−14.6, −10.0] | −27.4 [−29.9, −24.9] | 38.4% |
| Random | −169.4 [−171.9, −166.8] | −184.4 [−187.6, −181.2] | – |

Three comparisons, each an ablation of one piece:

- **Order flow is worth +$27.4** a game over inference from your hand alone.
- **Summing over the joint deal exactly is worth +$12.6** over treating opponents' hands
  as independent.
- **The marginal-value curve costs $30.8** against a flat value of pot/n per goal card.

Both tuned heuristics land near break-even. Beating them takes the whole inference stack.

### 2. Against three full Bayesian agents

| Focal agent | Chips per game | vs full Bayes, same deals | Goal suit correct |
|---|---|---|---|
| **Bayes, flat card value** | **+27.7** [+26.2, +29.1] | **+27.9** [+25.9, +30.0] | 76.1% |
| Bayes, mean-field order flow | +13.2 [+10.5, +15.9] | +13.5 [+10.9, +16.1] | 76.6% |
| Bayes, full (marginal value) | −0.3 [−2.2, +1.7] | – | 74.4% |
| Bayes, hand only | −40.7 [−43.2, −38.3] | −40.5 [−43.1, −37.8] | 38.4% |
| Scarcity heuristic | −41.1 [−43.3, −38.9] | −40.8 [−43.5, −38.2] | – |
| Long-suit heuristic | −58.5 [−60.8, −56.3] | −58.3 [−60.9, −55.6] | – |
| Random | −173.1 [−176.2, −170.0] | −172.8 [−176.1, −169.6] | – |

A full Bayesian agent among three copies of itself makes −$0.3 [−2.2, +1.7], which is
zero within noise, as symmetry requires. (Seat effects were checked separately with
`experiments/audit_claims.py`: over 800 games with four identical agents, every seat's
mean profit sits within noise of zero and every game is exactly zero-sum.)

Against Bayesian opponents the mean-field agent out-earns the exact one while calling the
goal suit slightly *better* (76.6% vs 74.4%). Accuracy across conditions is confounded,
since a different focal agent produces a different game and therefore different evidence.
The honest reading is that beliefs are not the binding constraint here: valuation is.

![Profit per game](figures/profit.png)

### 3. Why the marginal-value curve loses money

300 of the same deals, broken down by `experiments/diagnose_valuation.py`, focal agent
against three full Bayesian opponents. A tenth of the sample above, and means without
intervals, so treat it as the shape of the difference rather than its exact size.

| Focal agent | Goal cards bought | Goal cards sold | Net on other suits | Wins the bonus |
|---|---|---|---|---|
| Bayes, marginal value | 1.90 at **$23.0** | 2.06 at $23.0 | −$3.4 | **40%** of games |
| Bayes, flat value | 1.56 at **$10.4** | 2.08 at $19.0 | **+$25.0** | 9% |

The curve-following agent wins the majority far more often and pays $23 a card to do it.
The flat valuer mostly declines the race: it buys at less than half the price, sells into
the players fighting for the bonus, and earns $25 a game on the suits nobody is fighting
over.

The curve answers "what is this card worth to me if nobody else reacts?" Every player
pricing off it bids for the same pivotal cards at the same moment, and the premium they
pay exceeds the bonus they are chasing. Against identical Bayesian opponents, **72% of
games end in a tie for most goal cards** (`experiments/audit_claims.py`), so the majority
is usually shared anyway.

The effect grows with the length of the game: the flat valuer's edge goes $24.3 → $31.8 →
$33.0 across 60, 120 and 240 turns. More trading means more chances to overpay.

Valuing a card in the game where opponents are also bidding for the same threshold is the
fix, and it is the same equilibrium rabbit hole as a level-1 opponent model, so I stopped
here. I have also not isolated where the flat agent's other-suit profit comes from; it is
measured, not explained.

### 4. Order flow moves beliefs; your hand alone can't

![Belief convergence](figures/convergence.png)

Every agent starts at 0.291 on the true goal suit, where 0.25 is no information. Hand-only
inference stays there for the whole game: a dealt hand says what it says. With order flow
the belief climbs to 0.622 by the end, or 0.694 for the flat-value agent, which trades
more and therefore sees more. Mean-field reaches 0.434.

The trade-count axis mixes two effects: it counts the agent's own trades as well as
everyone else's, and trades carry exact constraints (nobody can sell a card they do not
hold) on top of behavioural signal. The curve stops at the trade count a quarter of games
still reach, so the tail is not a biased handful of busy games.

### 5. Hiding what you believe

The hypothesis was that buying your goal suit loudly teaches Bayesian opponents what it
is, so concealing your beliefs should pay. Both variants play three full Bayesian
opponents:

- **Slowed buying:** at most one buy every five turns in any suit it favours.
- **Decoy bids:** $6 bids in suits it thinks are unlikely to be the goal, 9.3 per game.

| Variant | Chips per game vs full Bayes | Change in opponents' final P(true goal) |
|---|---|---|
| Slowed buying | **+2.2** [+0.8, +3.6] | +0.005 [−0.003, +0.013] |
| Decoy bids | −32.8 [−35.4, −30.2] | **+0.082** [+0.065, +0.099] |
| Slowed + decoys | −31.0 [−33.6, −28.4] | +0.089 [+0.072, +0.106] |

Slowing down pays about two chips a game, but **not by leaking less**: opponents end up
exactly as well informed either way. Patience gets better prices, which is a different
mechanism than the one I set out to test.

Decoys backfire outright. Bayesian opponents value unlikely suits at a dollar or two, so
they hit a $6 bid at once and the decoy agent buys junk. Every one of those trades is a
hard constraint on somebody's hand, which makes opponents *more* accurate (+0.082) and the
decoy agent itself more accurate too (81.0% against 74.4%). A decoy is only free if nobody
fills it, and an order you never intend to have filled is a spoof.

![Information leakage](figures/leakage.png)

---

## The game

Forty cards: one suit has 12, two have 10, one has 8. The goal suit is the same-colour
partner of the 12-card suit, so it has 8 or 10 cards. Four players are dealt ten cards
each and ante $50 into a $200 pot. They then trade single cards for four minutes. At the
end, each goal-suit card pays $10, and whoever holds the most goal cards takes the rest of
the pot ($120 or $100). Ties split it.

There are exactly **twelve deck configurations**. Pick the 12-card suit (4 ways); its
partner is the goal. Then the goal has 8 and the other two suits have 10, or the goal has
10 and the other two are 10 and 8 (2 ways).

## How it works

### Engine and the agent boundary

```
figgie/
  deck.py        configurations, the 286 possible 10-card suit counts, dealing
  market.py      one continuous double auction per suit: limit orders, cancels,
                 price-time priority, trades at the resting price, self-trade prevention
                 (a trade then clears every book, as in real Figgie)
  protocol.py    the entire agent interface: Observation in, Action out, public events
  engine.py      turns, funding checks, settlement; owns every piece of hidden state
  inference.py   posterior over the 12 configurations (hand + order flow)
  valuation.py   majority-bonus share and the marginal value of a card
  agents/        random, two heuristics, Bayesian, concealing Bayesian
  tournament.py  paired-seed runs across processes, confidence intervals
experiments/     model fit, tuning, experiments, diagnostics, figures
```

An agent implements one method, `act(observation) -> action`. The observation holds only
your own current and dealt hands and chips, plus the attributed order books and the public
tape of placements, cancels and trades. It is a frozen dataclass, and the tape is a
read-only view. The deck configuration and other players' hands never cross the boundary.
A test walks every field of real observations and fails if anything else appears.

Four minutes of trading are split into 120 turns. Each turn every player acts once, in a
freshly shuffled order, so a game is a deterministic function of its seed. Resting orders
lock the chips or card behind them, so a resting order can always be filled.

As in real Figgie, a trade cancels every resting order in every suit, so players have to
requote after each print. Without that rule the book settles into a spread nobody crosses
and trading stops early.

### Inference

**Hand.** Your ten cards are a draw without replacement from a deck whose composition
depends on the configuration. P(hand | config) is a multivariate hypergeometric: twelve
exact numbers, normalised under a uniform prior. This already gives a distribution over
which suit is the goal *and* how many cards it has. Holding 5♠ 2♣ 2♥ 1♦ makes ♣ the goal
with probability 0.49 and ♦ only 0.19.

**Order flow.** Each placement by an opponent is a noisy signal about their dealt hand.
A **level-0 opponent model** turns it into a likelihood over the 286 hands they could
hold. The model assumes an opponent chooses which suit to bid on by softmax(β · P(goal = s
| their own hand)), and which suit to offer by the reverse. Bid price sharpens the choice,
and a fraction ε of placements are treated as noise. Trades also add hard, model-free
constraints: nobody can sell a card they don't hold.

**The joint deal, exactly.** Opponents' hands are not independent: all three come from
the same 30 unseen cards. Summing over the joint deal looks like 286² terms per
configuration. But the deal probability factorises as ∏ₛ rₛ! / (h_Bₛ! h_Cₛ! h_Dₛ!), so the
sum is a convolution over suit-count compositions. One `bincount` pairs two opponents
into a 20-card composition, and a lookup dots it with the third. That is about 1 ms per
update, and it matches brute-force enumeration to 1e-9 in the tests.

Treating opponents as independent instead (mean-field) is the obvious shortcut, and on
synthetic data it is badly wrong. On 1,500 placements drawn from the model itself, exact
inference put 0.997 on the true goal and mean-field put 0.63 (the scenario in
`test_mean_field_is_measurably_worse_than_the_exact_joint`). It also misses constraints:
two opponents who each sold six spades together prove spades is the 12-card suit, but
either one alone fits a 10-spade deck.

**Fitting the opponent model.** β and ε are maximum-likelihood estimates. Every placement
in 400 simulated games (83,843 placements) is scored against the placer's *true* dealt
hand, which only the harness knows after the game.

| Opponent model | Best fit | Information per placement vs uniform |
|---|---|---|
| "Bids on suits its own hand says are the goal" | β = 4, ε = 0.3 | **+0.054 bits** |
| "Bids on whatever it holds least of" | β = 0.1, ε = 0.7 | −0.033 bits (worse than uniform at every setting) |

A placement is worth little on its own; a game supplies about 210 of them.

The "bid what you hold least of" model is intuitive but wrong about this game. A long suit
is evidence that suit has 12 cards, which makes its *partner* the goal. Scarcity in your
hand says little about which suit is the goal.

### Valuation

If you hold k goal cards, your payout is 10k plus the bonus times your expected share of
it. That share is 1 if you hold strictly the most and 1/(t+1) if you tie with t others.
Opponents' holdings come from the posterior over their hands, conditioned on the exact
number of goal cards not in your hand.

The value of buying one card is $10 plus the bonus times the change in expected share,
assuming the card comes from an opponent in proportion to their holdings. The value of
selling one is the reverse. Under the dealing prior, buying from holdings is the same as
re-conditioning on one fewer card left for opponents (a tested identity). The expected
share over all deals comes to exactly 1/4 (also tested).

The **bid ceiling** for suit s is Σ over configurations with goal s of P(config) ×
marginal value. The **offer floor** is the same sum for losing a card.

All non-random agents share one execution layer: cancel quotes that have gone bad, take
anything priced better than value by at least $1, then quote at value minus or plus an
edge. So the agents differ only in what they believe cards are worth.

Quotes carry a few dollars of noise, shaved off the edge rather than added to it, so a
quote is never worse than the agent's own value. Without it every agent with the same
belief quotes the same price, and the market has nothing left to trade on.

### Agents

| Agent | Beliefs | Card value |
|---|---|---|
| `random` | none | none (always-legal random orders) |
| `scarcity` | goal = suit dealt fewest of | flat, tuned |
| `long_suit` | goal = partner of suit dealt most of | flat, tuned |
| `bayes_hand_only` | exact hand posterior | inventory-aware |
| `bayes_mean_field` | hand + order flow, opponents independent | inventory-aware |
| `bayes_flat_value` | hand + order flow, exact joint | flat pot / n per goal card |
| `bayes` | hand + order flow, exact joint | inventory-aware |
| `conceal_*` | as `bayes`, plus slowed buying and/or decoy bids | inventory-aware |

The benchmarks are tuned so they aren't strawmen. Every family is tuned over the **same**
knobs by coordinate search: sweep one parameter with the others fixed, keep the best,
repeat (70 configurations, 1,500 paired games each, seeds disjoint from evaluation).

Tuning one parameter for one family and not another quietly hands that family an edge. An
earlier version of this swept the minimum offer price for the Bayesian agent alone and
left the heuristics at the default, which flattered the Bayesian agent's headline number.

| Family | Tuned to |
|---|---|
| `scarcity` | goal value $4, other suits $4, minimum offer $20 |
| `long_suit` | goal value $8, other suits $4, minimum offer $20 |
| `bayes` | minimum offer $25 |

Every family lands on a high minimum offer, and that is the single most valuable knob for
all of them: not giving away unwanted cards beats being aggressive about the goal suit. A
$25 floor means the Bayesian agent almost never offers, since a goal card is worth at most
about $25 — which is really the tuner telling me the quoting logic gives away edge.

## Where I stopped, and why

- **Level-0 opponents.** The opponent model assumes other players act on their own hands
  and ignore everyone else's orders, which isn't true of the Bayesian agents. A level-1
  model, where opponents reason about what *you* have revealed, means simulating their
  posterior over your hand inside yours. I didn't build it. The leakage experiment found
  no measurable cost to revealing beliefs against these opponents, and more accurate
  beliefs didn't earn more against Bayesians (mean-field vs exact). Both suggest the next
  dollar is in valuation, not deeper opponent modelling.
- **Myopic valuation, and it is the main finding.** Marginal value is computed at current
  holdings, as if opponents will not react. Section 3 shows that costs $30.8 a game
  against heuristics and $27.9 against Bayesians. Valuing cards in equilibrium is the
  obvious next piece of work, and the one I would do first.
- **Price enters the likelihood as a sharpness multiplier.** It is not a generative model
  of prices.
- **Decoy bids are close to spoofing.** Resting bids you'd rather not have filled are
  illegal in real markets, and they lost money here anyway. "Slowed buying" is the
  legitimate version of concealment.
- **The market thins out before the end.** The median last trade is turn 50 of 120 and
  about 30% of games still trade after turn 60, at about 29 trades a game. An earlier version
  without book clearing or quote noise died at turn 16, and every conclusion here got
  sharper once the market stayed alive, so the remaining quiet tail is worth treating as a
  limitation rather than a settled market. Order flow is worth $27.8-$29.7 at 60, 120 and
  240 turns (`experiments/horizon_check.py`), but the flat valuer's edge keeps growing
  with the horizon, so that comparison is not horizon-free.
- **Placements are treated as independent evidence.** A game produces about 210 of them,
  many being requotes of an unchanged view after a trade clears the book. The likelihood
  multiplies them as if each were a fresh draw, which ought to leave the posterior
  overconfident. Measured over 800 games it is not: stated confidence matches how often
  the agent is right to within 0.001 (`experiments/audit_claims.py`). Only the top bin is
  optimistic, saying 0.99 where it is right 0.94. The mechanism is still wrong even though
  the symptom does not show up at this scale.
- **Single-card orders; integer chips.** A bonus that can't split evenly gives its odd
  chips to tied winners in seat order. Seats rotate across seeds, so no seat is favoured.

## Tests

`pytest` runs 108 tests in about 10 seconds:

- **Conservation.** Cards and chips are conserved after *every* action of 20 random-agent
  games. Books are never crossed, and locked chips and cards always equal what the resting
  orders need.
- **Settlement.** Payouts always sum to exactly the pot, including three-way ties.
- **Matching.** Better price fills first. Earlier order fills first at equal price. Trades
  print at the resting price. Immediate-or-cancel orders never rest. Self-trade prevention
  works.
- **Engine/agent boundary.** Observations contain only whitelisted public types. Illegal
  actions are rejected. The same seed gives the same game.
- **Hand inference.** The hand likelihood matches scipy's multivariate hypergeometric and
  sums to 1 over all 286 hands. The posterior sums to 1, rules out impossible
  configurations, and is calibrated over 20,000 deals.
- **Order-flow inference.** The posterior converges to the true configuration on synthetic
  order flow. The convolution matches brute-force enumeration of the joint deal, for both
  posteriors and hand marginals. Mean-field is measurably worse. Selling constraints across
  opponents are exact. The truth is never ruled out in real games.
- **Valuation.** Expected share over all deals is 1/4. Buying from holdings matches
  re-conditioning. The marginal value curve is hump-shaped and lies between $10 and
  $10 + bonus.
- **Card accounting.** At points throughout a real game, the goal cards an agent holds
  plus the goal cards it believes opponents hold must equal the size of the suit, and no
  configuration it still believes in may be left with no consistent opponent holdings.
  This one is here because it failed. Beliefs used to clip impossible holdings up to zero,
  which inflated opponents' assumed cards and mostly hit the ablation baselines, so the
  headline numbers measured against them were flattered: order flow scored 42.9 chips a
  game before the fix and 27.4 after, the exact joint 18.1 before and 12.6 after.

## Reproduce

```bash
pip install -e ".[test]"
experiments/reproduce.sh
```

`experiments/reproduce.sh` runs, in order: tests, the opponent-model fit, benchmark
tuning, the three paired experiments (3,000 seeds each), the valuation diagnostic, the
horizon check and the figures. Results land in `results/*.json` and figures in `figures/`.

The whole pipeline plays about 208,000 games and takes roughly an hour on ten cores.
Tuning is most of it (144,000 games); the headline experiment alone is four minutes. A
Bayesian decision costs about a millisecond, almost all of it the 286x286 pairing behind
the joint deal.

### Checking the numbers

`experiments/verify_claims.py` runs last and checks all 35 figures quoted in this README
against `results/*.json`, exiting non-zero if any of them no longer matches what the code
produces. It also fails if chips stop being conserved across players, or if any seat shows
a profit outside noise. A claim here cannot quietly drift away from its evidence.

A game is a deterministic function of its seed, so a clone reproduces these figures
exactly on the same library versions. The ones used are recorded in `results/audit.json`
(Python 3.13.5, numpy 2.5.3, macOS); a different numpy could in principle shift a
floating-point comparison and change a decision.
