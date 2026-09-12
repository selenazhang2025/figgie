"""Checks every number published in the README against results/*.json.

Run it after reproduce.sh. It fails loudly if any published figure no longer matches
what the code produces, so a claim can't quietly drift away from its evidence.
"""

from __future__ import annotations

import sys

from common import read_json

TOL = 0.06  # chips, unless a claim says otherwise


def main() -> int:
    head, bf = read_json("headline.json"), read_json("bayes_field.json")
    leak, fit = read_json("leakage.json"), read_json("opponent_model_fit.json")
    tune, mech = read_json("heuristic_tuning.json"), read_json("valuation_mechanism.json")
    hz, audit = read_json("horizon_check.json"), read_json("audit.json")

    def profit(d, name, field="profit"):
        return d["conditions"][name][field]["mean"]

    def acc(d, name):
        return d["conditions"][name]["belief"]["accuracy_end"]["mean"]

    claims: list[tuple[str, float, float, float]] = [
        # headline table
        ("headline flat-value profit", profit(head, "bayes_flat_value"), 45.9, TOL),
        ("headline full profit", profit(head, "bayes"), 15.1, TOL),
        ("headline mean-field profit", profit(head, "bayes_mean_field"), 2.5, TOL),
        ("headline scarcity profit", profit(head, "scarcity"), -1.0, TOL),
        ("headline long-suit profit", profit(head, "long_suit"), -1.9, TOL),
        ("headline hand-only profit", profit(head, "bayes_hand_only"), -12.3, TOL),
        ("headline random profit", profit(head, "random"), -169.4, TOL),
        # headline paired differences
        ("flat value beats marginal value", profit(head, "bayes_flat_value", "vs_bayes"), 30.8, TOL),
        ("order flow is worth", -profit(head, "bayes_hand_only", "vs_bayes"), 27.4, TOL),
        ("exact joint is worth", -profit(head, "bayes_mean_field", "vs_bayes"), 12.6, TOL),
        ("RESUME flat value beats best benchmark", profit(head, "bayes_flat_value", "vs_scarcity"), 46.9, TOL),
        # accuracy
        ("RESUME flat-value accuracy", acc(head, "bayes_flat_value"), 0.782, 0.001),
        ("RESUME hand-only accuracy", acc(head, "bayes_hand_only"), 0.384, 0.001),
        ("headline full accuracy", acc(head, "bayes"), 0.709, 0.001),
        # against three Bayesians
        ("RESUME flat value beats marginal value, all-Bayes table",
         profit(bf, "bayes_flat_value", "vs_bayes"), 27.9, TOL),
        ("all-Bayes symmetry check (must be ~0)", profit(bf, "bayes"), -0.3, 2.0),
        # concealment
        ("RESUME decoy orders cost", profit(leak, "conceal_decoy", "vs_bayes"), -32.8, TOL),
        ("slowed buying earns", profit(leak, "conceal_slow", "vs_bayes"), 2.2, TOL),
        ("decoys leave opponents no better informed",
         leak["conditions"]["conceal_slow"]["field_p_true_end_vs_bayes"]["mean"], 0.005, 0.001),
        ("decoys make opponents more accurate",
         leak["conditions"]["conceal_decoy"]["field_p_true_end_vs_bayes"]["mean"], 0.082, 0.001),
        ("decoys per game", leak["conditions"]["conceal_decoy"]["decoys_per_game"], 9.3, 0.05),
        # opponent model fit
        ("RESUME observed orders", fit["placements"], 83843, 0),
        ("RESUME scarcity model, bits per order",
         max(r["bits_per_placement"] for r in fit["grid"] if r["kind"] == "scarcity"), -0.033, 0.0005),
        ("fitted model, bits per order",
         max(r["bits_per_placement"] for r in fit["grid"] if r["kind"] == "posterior"), 0.054, 0.0005),
        # valuation diagnostic
        ("marginal-value agent pays per goal card", mech["conditions"]["bayes"]["goal_buy_price"], 23.0, TOL),
        ("flat-value agent pays per goal card", mech["conditions"]["bayes_flat_value"]["goal_buy_price"], 10.4, TOL),
        ("marginal-value agent wins the bonus", mech["conditions"]["bayes"]["bonus_won_rate"], 0.40, 0.01),
        ("flat-value agent wins the bonus", mech["conditions"]["bayes_flat_value"]["bonus_won_rate"], 0.09, 0.01),
        ("flat-value agent, other suits", mech["conditions"]["bayes_flat_value"]["other_suits_net_per_game"], 25.0, TOL),
        # market and audit
        ("trades per game", hz["quiescence"]["trades_per_game"], 29.0, 0.05),
        ("median last trade turn", hz["quiescence"]["median_last_trade_turn"], 50, 0.5),
        ("games ending in a tie for the majority", audit["tie_rate"], 0.722, 0.005),
        ("posterior overconfidence", audit["calibration"]["overconfidence"], 0.0, 0.02),
        ("tuning configurations searched", len(tune["rows"]), 70, 0),
        ("paired deals per condition", head["n_seeds"], 3000, 0),
    ]

    failures = []
    for label, got, want, tol in claims:
        ok = abs(got - want) <= tol
        if not ok:
            failures.append(f"{label}: published {want}, code says {got:.4f}")
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<48} {want:>10}  (code: {got:.4f})")

    if not audit["symmetry"]["every_game_zero_sum"]:
        failures.append("chips are not conserved across players")
    for seat, (mean, half) in enumerate(zip(audit["symmetry"]["seat_mean_profit"], audit["symmetry"]["ci_half_width"])):
        if abs(mean) > half:
            failures.append(f"seat {seat} profit {mean:+.2f} is outside noise (+/-{half:.2f}): seat bias")

    print()
    if failures:
        print(f"{len(failures)} published claim(s) no longer match the code:")
        print("\n".join("  " + f for f in failures))
        return 1
    print(f"all {len(claims)} published numbers match results/*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
