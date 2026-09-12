#!/usr/bin/env bash
# Rebuild every number and figure in the README from scratch.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/experiments"
PY="${PYTHON:-python}"

"$PY" -m pytest -q
"$PY" experiments/calibrate_opponent_model.py --games 400
"$PY" experiments/tune_heuristics.py --games 1500
"$PY" experiments/run_experiments.py headline bayes_field leakage --seeds 3000
"$PY" experiments/diagnose_valuation.py --seeds 300
"$PY" experiments/horizon_check.py --seeds 800
"$PY" experiments/make_figures.py
