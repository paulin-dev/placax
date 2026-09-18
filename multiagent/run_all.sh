#!/usr/bin/env bash
# Re-runs every experiment report.html is built from, into the directory names build_report.py
# reads, then rebuilds the report.
#
#     bash multiagent/run_all.sh            # from the repository root; ~45 min on a laptop GPU
#     PY=python bash multiagent/run_all.sh  # choose the interpreter (default: venv/bin/python)
#
# Idempotent: a run whose summary.json already exists is skipped, so an interrupted run_all
# resumes where it stopped. Delete a directory under multiagent/runs/ to redo just that run.
#
# Expect the same conclusions, not the same digits. GPU reductions are not bit-deterministic, the
# first sweep was originally run before `policy.make_act` existed (it splits the random key
# differently), and its transfers were first scored by the original legalizer; everything is
# scored by the final one here.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-venv/bin/python}
R=multiagent/runs
LOG=$R/run_all.log
mkdir -p "$R"
echo "run_all started $(date)" >> "$LOG"

canvas_of() { [ "$1" = ariane133 ] && echo die || echo core; }
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Runs a python module into an output directory unless that directory already has results.
# usage: once <dir> <done-file> <module> <args...>
once() {
  local dir=$1 done_file=$2; shift 2
  if [ -f "$dir/$done_file" ]; then return; fi
  say "$*"
  mkdir -p "$(dirname "$dir")"
  "$PY" -m "$@" >> "$LOG" 2>&1
}

TRAIN="--benchmark_dir=benchmarks/adaptec1 --iterations=100 --eval_every=5 --lr=1e-3"
ADAM="--steps=2000 --eval_every=100 --lr=0.05"

# --- Sweep 1: sight, big-then-small steps, overlap penalty (no penalty = the crowding result) ---
sweep1=(
  "m1|--view=m1"
  "m1all|--view=m1all"
  "m1all-sched|--view=m1all --max_step_start=16"
  "m1all-ov3|--view=m1all --overlap_weight=3"
  "m1all-ov10|--view=m1all --overlap_weight=10"
  "m1all-sched-ov10|--view=m1all --max_step_start=16 --overlap_weight=10"
  "m0-sched-ov10|--view=m0 --max_step_start=16 --overlap_weight=10"
)
for entry in "${sweep1[@]}"; do
  name=${entry%%|*}; flags=${entry#*|}
  for seed in 0 1 2; do
    d=$R/sweep/$name-s$seed
    once "$d" summary.json multiagent.train $TRAIN --seed=$seed --out=$d $flags
    once "$R/sweep/xfer-$name-s$seed" summary.json multiagent.transfer --run=$d \
      --benchmark_dir=benchmarks/bigblue1 --out=$R/sweep/xfer-$name-s$seed
  done
done
for ov in 0 3 10; do
  for chip in adaptec1 bigblue1; do
    once "$R/sweep/adam-ov$ov-$chip" summary.json multiagent.adam --benchmark_dir=benchmarks/$chip \
      $ADAM --overlap_weight=$ov --out=$R/sweep/adam-ov$ov-$chip
  done
done

# --- Sweep 2: everything at overlap penalty 3 - sight, schools, generalization ---
sweep2=(
  "m0|--view=m0"
  "m1|--view=m1"
  "m1all|--view=m1all"
  "align0.5|--view=m1all --align=0.5"
  "align0.9|--view=m1all --align=0.9"
  "align0.9-sched4|--view=m1all --align=0.9 --max_step_start=4"
  "noise4|--view=m1all --start_noise=4"
  "async0.5|--view=m1all --update_prob=0.5"
  "noise4-2chips|--view=m1all --start_noise=4 --extra_benchmarks benchmarks/bigblue1"
)
for entry in "${sweep2[@]}"; do
  name=${entry%%|*}; flags=${entry#*|}
  for seed in 0 1 2; do
    d=$R/sweep2/$name-s$seed
    once "$d" summary.json multiagent.train $TRAIN --overlap_weight=3 --seed=$seed --out=$d $flags
    # Transfers, scored by the final legalizer, each followed by the swap search.
    for chip in bigblue1 ariane133; do
      x=$R/q3/xfer-sweep2-$name-s$seed-$chip
      once "$x" summary.json multiagent.transfer --run=$d --benchmark_dir=benchmarks/$chip \
        --canvas=$(canvas_of $chip) --out=$x
      once "$x/swap" summary.json multiagent.swap --benchmark_dir=benchmarks/$chip \
        --canvas=$(canvas_of $chip) --positions=$x/positions.npy --out=$x/swap
    done
  done
done
for ov in 0 3; do
  once "$R/sweep2/adam-ov$ov-ariane133" summary.json multiagent.adam --benchmark_dir=benchmarks/ariane133 \
    --canvas=die $ADAM --overlap_weight=$ov --out=$R/sweep2/adam-ov$ov-ariane133
done

# --- The swap search from the greedy start ---
for chip in adaptec1 bigblue1 ariane133; do
  once "$R/q3/swap-$chip" summary.json multiagent.swap --benchmark_dir=benchmarks/$chip \
    --canvas=$(canvas_of $chip) --out=$R/q3/swap-$chip
done

# --- The swap swarm: exact judge, learned judges, nudge-then-swap ---
W=$R/swarmswap
for chip in adaptec1 bigblue1 ariane133; do
  c=$(canvas_of $chip)
  once "$W/exact-$chip-k0" summary.json multiagent.swarm_swap run --benchmark_dir=benchmarks/$chip \
    --canvas=$c --k=0 --max_rounds=200 --out=$W/exact-$chip-k0
  for k in 0 8; do
    once "$W/exact-$chip-k$k-resolve" summary.json multiagent.swarm_swap run --benchmark_dir=benchmarks/$chip \
      --canvas=$c --k=$k --resolve --max_rounds=200 --out=$W/exact-$chip-k$k-resolve
  done
  once "$W/cycles-$chip-c1-nudgefirst" summary.json multiagent.swarm_swap run --policy=$R/sweep2/m1-s0 \
    --benchmark_dir=benchmarks/$chip --canvas=$c --k=0 --resolve --nudge_first --max_rounds=200 \
    --out=$W/cycles-$chip-c1-nudgefirst
done
for view in m0 m1 m1all; do
  for seed in 0 1 2; do
    once "$W/net-$view-s$seed" scorer.json multiagent.swarm_swap train --view=$view --seed=$seed \
      --out=$W/net-$view-s$seed
    for chip in adaptec1 bigblue1 ariane133; do
      once "$W/learned-$view-s$seed-$chip-k0" summary.json multiagent.swarm_swap run \
        --scorer=$W/net-$view-s$seed --benchmark_dir=benchmarks/$chip --canvas=$(canvas_of $chip) \
        --k=0 --resolve --threshold=0.3 --max_rounds=100 --out=$W/learned-$view-s$seed-$chip-k0
    done
  done
done
for view in m1 m1all; do
  for seed in 0 1 2; do
    once "$W/net2-$view-s$seed" scorer.json multiagent.swarm_swap train --view=$view --seed=$seed \
      --extra_benchmarks benchmarks/bigblue1 --out=$W/net2-$view-s$seed
    once "$W/learned2-$view-s$seed-ariane133" summary.json multiagent.swarm_swap run \
      --scorer=$W/net2-$view-s$seed --benchmark_dir=benchmarks/ariane133 --canvas=die \
      --k=0 --resolve --threshold=0.3 --max_rounds=100 --out=$W/learned2-$view-s$seed-ariane133
  done
done

# --- Pictures the report embeds ---
[ -f "$R/sweep/m1-s0/viz/before_after.png" ] || { say "visualize sweep/m1-s0";
  "$PY" -m multiagent.visualize --run=$R/sweep/m1-s0 --only before_after >> "$LOG" 2>&1; }
[ -f "$R/sweep2/m1all-s0/viz/episode.gif" ] || { say "visualize sweep2/m1all-s0";
  "$PY" -m multiagent.visualize --run=$R/sweep2/m1all-s0 --only episode before_after >> "$LOG" 2>&1; }
mkdir -p "$W/gifs"
gif() { # <file> <args...>
  local file=$W/gifs/$1; shift
  [ -f "$file" ] || { say "gif $file"; "$PY" -m multiagent.swarm_swap run "$@" --gif="$file" >> "$LOG" 2>&1; }
}
gif ariane133-resolve.gif --benchmark_dir=benchmarks/ariane133 --canvas=die --k=0 --resolve --max_rounds=200
gif ariane133-k8.gif --benchmark_dir=benchmarks/ariane133 --canvas=die --k=8 --resolve --max_rounds=200
gif bigblue1-resolve.gif --benchmark_dir=benchmarks/bigblue1 --k=0 --resolve --max_rounds=200
gif adaptec1-all-at-once.gif --benchmark_dir=benchmarks/adaptec1 --k=0
gif adaptec1-resolve.gif --benchmark_dir=benchmarks/adaptec1 --k=0 --resolve
gif adaptec1-nudge-then-swap.gif --policy=$R/sweep2/m1-s0 --benchmark_dir=benchmarks/adaptec1 --k=0 --resolve --nudge_first

# --- The report: re-score everything under the final legalizer, then fill the template ---
say "build_report --recompute"
"$PY" -m multiagent.build_report --recompute 2>&1 | tee -a "$LOG"
say "done"
