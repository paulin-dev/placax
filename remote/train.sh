#!/usr/bin/env bash
# Runs the MaskPlace pipeline end to end on a rented GPU box (e.g. vast.ai): finds the
# largest n_episodes that fits, trains, then visualizes. Mirrors colab/run_maskplace.ipynb,
# minus the Colab-only bits (Drive mount, preinstalled-JAX reuse).
#
# Run this inside tmux/screen so training survives an SSH disconnect:
#   tmux new -s train
#   ./remote/train.sh
#   # Ctrl-b d to detach, `tmux attach -t train` to check back in
#
# Usage: ./remote/train.sh [benchmark_dir] [n_iterations]
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

BENCHMARK_DIR="${1:-benchmarks/adaptec1}"
N_ITERATIONS="${2:-300}"
MACRO_BUDGET="${MACRO_BUDGET:-128}"

echo "== searching for the largest n_episodes that fits =="
python -m scripts.subprocess_search scripts.run_maskplace '--n_episodes=[1,2,4,8,10]' \
    --benchmark_dir="$BENCHMARK_DIR" --macro_budget="$MACRO_BUDGET" --eval_every=1 --n_iterations=4 --no_checkpoint \
    2>&1 | tee /tmp/n_episodes_search.log

N_EPISODES=$(grep -oP 'RESULT=\K.*' /tmp/n_episodes_search.log)
echo "-> n_episodes=${N_EPISODES}"

echo "== training (re-run this script to resume from checkpoint) =="
python -m scripts.run_maskplace --benchmark_dir="$BENCHMARK_DIR" --macro_budget="$MACRO_BUDGET" \
    --n_episodes="$N_EPISODES" --n_iterations="$N_ITERATIONS" --patience=10

echo "== visualizing =="
python -m scripts.visualize --benchmark_dir="$BENCHMARK_DIR" --macro_budget="$MACRO_BUDGET" --preset=maskplace

echo "Done. Pull results back from your local machine with, e.g.:"
echo "  rsync -avz -e 'ssh -p <port>' <user>@<host>:placax/${BENCHMARK_DIR}/output_maskplace/ ./output_maskplace/"
