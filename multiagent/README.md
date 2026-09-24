# multiagent — every macro moves itself

An experiment kept outside `placax/` and `placax_agents/`. It **imports** the core (netlist loading,
smoothed wirelength, density cost, greedy warm start, legality measurement, `score_placement`) and
adds what the core can't express: an action that moves **every macro at once**, one small step
each, chosen by **one shared policy applied per macro**, trained by differentiating the whole-chip
score through the episode.

> Can a placement emerge from many identical, locally informed macros cooperating on one global
> score — and does it beat optimizing the positions directly?

**The full write-up is the paper, [`paper/paper.pdf`](paper/paper.pdf)** (NeurIPS-style, 14 pages),
with its animations in [`paper/SUPPLEMENTARY.md`](paper/SUPPLEMENTARY.md). This README is the short
version and the how-to. For the same results explained from zero, with every figure and animation
described in words, open [`explainer/macro-swarm-explained.html`](explainer/macro-swarm-explained.html)
in a browser.

## Results in one table

adaptec1 (128 macros, 224 grid, core canvas), starting from the greedy-wiremask placement
(441,257 HPWL). Numbers are legal HPWL improvement over that start, after `legalize.py`. Learned
runs are the **median of the last 5 checkpoints, averaged over 3 seeds**, never the best
checkpoint.

| method | alone | then `swap.py` |
|---|---|---|
| random jitter (1 cell) + legalize — the no-intelligence control | −2.6% | — |
| `swap.py` — swap same-size macros, no learning, 0.5 s | +18.1% | — |
| `anneal.py` — simulated annealing, 30 s / 600 s | +19.2% / +25.7% | — |
| `pull.py` — step toward your partners, no learning | +11.0% | +19.8% |
| `adam.py`, overlap penalty 3 (the agents' objective) | +2.0% | +20.0% |
| `adam.py`, its best penalty (0) | +16.7% | +17.0% |
| **agents** m1all / m1, overlap penalty 3 | **+7.0%** / +5.7% | +24.2% / **+25.0%** |

On chips the policy never saw, a swap search alone wins: bigblue1 +44.4% and ariane133 +33.8%,
against at best +8.6% and −1% for transferred policies. On bigblue1, random jitter alone ranges
from −8.3% to +8.3%, so a transfer gain has to clear that band before it counts.

| question | answer |
|---|---|
| Q1 — can local agents with one shared brain improve a placement? | **Yes, modestly and reliably:** +7%, seeds within 0.2 points |
| Q2 — how much must a macro see? | **Itself, on the chip it trained on (it memorizes); its partners, for transfer** (m0 −1.8% vs m1 +7.5% on bigblue1) |
| Q3 — better than direct optimization? | **Mixed.** Yes on the same objective; no against Adam at its best penalty or against swaps; yes as the refinement step before swaps (+25%) |
| Q4 — transfer to an unseen chip? | **No.** Inside the jitter band on bigblue1, negative on ariane133 |
| swarm rules (boids alignment, schools) | **No.** Partners moving together can't shorten the wires between them |
| agents that swap | **Yes, with an exact local judge:** a leaderless swarm matches or beats the centralized swap search. **No, with a learned one:** it doesn't transfer |

### Agents that swap (`swarm_swap.py`)

Every macro considers trading places with its `k` nearest same-footprint macros; a swap happens
when two macros pick each other, and all agreed swaps in a round happen at once. No coordinator.

| | adaptec1 | bigblue1 | ariane133 |
|---|---|---|---|
| centralized swap search (`swap.py`) | +18.1% | +44.4% | +33.8% |
| swarm, exact local gain, all swaps at once | **−15.1%** | +38.5% | +37.3% |
| swarm, exact local gain, **wired swaps wait** (`--resolve`) | **+18.1%** | **+44.4%** | **+35.3%** |
| same, but only the 8 nearest candidates | +18.1% | +27.8% | +15.5% |
| swarm, learned judge (m1/m1all view), trained on adaptec1 | +18% on 1–2 of 3 seeds | −38% to 0% | −1% to +1% |
| same, trained on adaptec1 + bigblue1 | — | *(trained on)* | −0.1% to +2.4% |
| learned judge, m0 view | 0% | 0% | 0% |
| nudging policy (m1, penalty 3, one seed), then swap swarm | **+25.8%** | +38.8% | +27.8% |

- **A leaderless swap swarm matches the centralized search, and beats it on ariane**, in fewer
  parallel rounds than the search's sequential swaps. But only with **local conflict
  resolution**: a swap waits if a wired partner is in a better one this round. Without it,
  swaps that share nets interfere, and adaptec1 gets 15% worse.
- **Swap decisions need a wide candidate range.** Restricting a macro to its 8 nearest
  same-size macros halves the gain on ariane.
- **Learning when to swap from the nudging policy's view does not transfer**, even from two
  chips. m0 never finds a useful swap: without its partners' positions, a macro has nothing to go
  on.
- **Nudges and swaps combine only on the chip the policy trained on** (+25.8% on adaptec1). On
  unseen chips the nudges undo part of what the swaps gained, and repeating the cycle makes it
  worse.

### What the numbers needed before they meant anything

- **An overlap penalty.** Without it, the agents crowd macros onto each other and the legalizer
  decides the result. Seeds disagreed by 26 points, and the best checkpoint overstated the median
  by ~15. Use `--overlap_weight=3`.
- **A random-jitter control.** On a sparse canvas (bigblue1 is 3.5% full), the wire-aware
  legalizer is a lottery: noise alone can gain 8%.
- **A legalizer that works on a packed canvas.** On ariane133 (identical SRAMs covering 50% of the
  canvas), the original repair turned a 1-cell overlap into a ~28-cell jump. `legalize.spread`
  pushes overlapping pairs apart first; `repair_and_report` tries both ways and keeps the shorter
  legal result.
- **A swap search.** Continuous moves can't make two blocks trade places without passing through
  each other. That is where most of the slack in these starts is.
- **Simulated annealing, given the same wall clock.** It makes the same swap move, so it is the
  baseline the swarm has to answer. At one second the swarm wins everywhere; at 600 s annealing
  wins on the dense design (+45.4% vs +35.3%). Our timing is implementation-bound: the swarm is
  GPU-batched, the annealer a single-threaded NumPy loop.
- **A temperature, once annealing had shown what was missing.** The swarm stops at the local
  optimum of its move set; annealing does not. `--anneal_seconds --descend_first` keeps consent and
  the conflict rule and lets each macro accept its own proposal by the Metropolis rule, starting
  from the swarm's own answer. From 120 s upward it is ahead of the sequential annealer on all
  three designs. It must propose a *random* same-size partner, not its best one, and every round is
  verified against the grid because simultaneous moves can otherwise overlap (one or two rounds a
  run are dropped).

## Quick start

```bash
# The shared policy, the method under test (~30 s on a laptop GPU)
python -m multiagent.train --benchmark_dir=benchmarks/adaptec1 --view=m1 --overlap_weight=3 --iterations=100 --lr=1e-3

# The same objective optimized directly on the positions
python -m multiagent.adam --benchmark_dir=benchmarks/adaptec1 --overlap_weight=3 --steps=2000 --eval_every=100 --lr=0.05

# The controls
python -m multiagent.pull --benchmark_dir=benchmarks/adaptec1 --steps 32
python -m multiagent.swap --benchmark_dir=benchmarks/adaptec1
python -m multiagent.swap --benchmark_dir=benchmarks/adaptec1 --positions=multiagent/runs/<run>/best_positions.npy

# Simulated annealing (the classic method built on the same swap move), and wall-clock timing
python -m multiagent.anneal --benchmark_dir=benchmarks/adaptec1 --seconds=30 --seeds=3
python -m multiagent.timing --out=multiagent/runs/timing

# The swap swarm: exact judge, and a learned one (train on adaptec1, apply anywhere)
python -m multiagent.swarm_swap run --benchmark_dir=benchmarks/ariane133 --canvas=die --k=0 --resolve
python -m multiagent.swarm_swap train --view=m1all --seed=0 --out=multiagent/runs/<scorer>
python -m multiagent.swarm_swap run --scorer=multiagent/runs/<scorer> --benchmark_dir=benchmarks/bigblue1 --k=0 --resolve --threshold=0.3
python -m multiagent.swarm_swap run --policy=multiagent/runs/<policy run> --k=0 --resolve --nudge_first

# The parallel annealer: the swap swarm with a temperature, started from the swarm's own answer
python -m multiagent.swarm_swap run --benchmark_dir=benchmarks/adaptec1 --k=0 --resolve \
  --anneal_seconds=120 --descend_first --proposal=random --seed=0 --out=multiagent/runs/swarmanneal/<name>

# A trained policy on a design it never saw (ariane133 has no rows: --canvas=die)
python -m multiagent.transfer --run=multiagent/runs/<run> --benchmark_dir=benchmarks/bigblue1

# Watch it: episode.gif, training.gif, before_after.png, curves.png into <run>/viz/
python -m multiagent.visualize --run=multiagent/runs/<run> --against multiagent/runs/<adam run>

# All runs as one table
python -m multiagent.compare multiagent/runs/*

# Re-run every experiment the paper is built from (~45 min), then rebuild it
bash multiagent/run_all.sh

# Rebuild the paper from the runs: figures, tables, quoted numbers, then LaTeX (--recompute after new runs)
# (needs Tectonic, a self-contained LaTeX engine, once: the static build in ~/.local/bin, e.g.
#  curl -fsSL https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz | tar -xz -C ~/.local/bin)
python -m multiagent.paper.build

# The plain-language version of the same results, as one self-contained HTML file
python -m multiagent.explainer.build

# Smoke tests (8 macros on a 32 grid), after every change
python -m pytest multiagent/test_multiagent.py -q
```

### `train.py` options that change the experiment

| flag | what it does | finding |
|---|---|---|
| `--view` | `m0` itself · `m1` + top-4 partners · `m1all` + all partners · `m2` + global summary | m0 = m1 = m1all on adaptec1; only partner views transfer |
| `--overlap_weight` | pairwise overlap penalty | 3 is the setting that made results reproducible |
| `--max_step_start` | step bound shrinking geometrically to `--max_step` | hurts: crushes everything, or scrambles past recovery |
| `--align` | boids alignment: blend your move with your partners' | hurts: +7% → ~0% |
| `--update_prob` | a random subset of macros moves each step (NCA-style) | best bigblue1 transfer (+8.6%), still inside the jitter band |
| `--start_noise` | train from jittered starts | less memorization, better bigblue1 transfer, worse home chip |
| `--extra_benchmarks` | train on several designs in turn | least-bad ariane transfer |

## The pieces

| file | what it is |
|---|---|
| `context.py` | one loaded design: greedy warm start, bounds, footprints, neighbour table |
| `neighbors.py` | who is wired to whom (clique weights, `2/m` per net), each macro's top-k partners |
| `objective.py` | the shared score: smoothed wirelength + density + pairwise overlap, all normalized |
| `moves.py` | the transition: every macro moves at once, clipped; gradient cut every `--horizon` |
| `view.py` | what one macro sees: `m0`, `m1`, `m1all`, `m2` |
| `policy.py` | the shared network, the step bound, and `make_act` — the one decision rule train, transfer and visualize all use (alignment, async updates) |
| `train.py` | short-horizon analytic-gradient training (SHAC's window, no critic) |
| `adam.py` | the same objective optimized directly on the positions |
| `pull.py` | untrained control: step toward the weighted centre of your partners |
| `swap.py` | swap same-footprint macros while it shortens the wires; legal by construction |
| `anneal.py` | simulated annealing on the same moves: the baseline the swap swarm has to answer |
| `timing.py` | wall clock of every method, compile time separated from steady state |
| `swarm_swap.py` | the swap swarm: local candidates, mutual consent, `--resolve`, exact or learned judge, `--policy` nudge/swap cycles |
| `legalize.py` | wire-aware repair, `spread` for packed canvases, and the both-ways portfolio |
| `transfer.py` | run a trained policy on another design, no retraining |
| `visualize.py` | the GIFs and plots for one run |
| `compare.py` | runs as one table, with an environment-mismatch warning |
| `results.py` | every number the paper reports, read from `runs/`; caches the three derived measurements in `runs/q3/` |
| `paper/` | `paper.tex` + `refs.bib` (the paper), `build.py` (figures, tables, `numbers.tex`, then Tectonic), `SUPPLEMENTARY.md` + `media/` (animations) |
| `explainer/` | `build.py` writes `macro-swarm-explained.html`: the same results explained from zero, one self-contained file with the animations embedded |
| `run_all.sh` | every experiment the paper reads, in order; skips runs that already exist |

Every run directory holds `manifest.json` (written before training), `log.jsonl`, `summary.json`,
`snapshots/iter_*.npy` (the raw placement at each evaluation — re-legalize these rather than
retraining), `best_positions.npy` (repaired, legal), `best_positions_raw.npy`, `best_params.pkl`
and `last_params.pkl`.

## How to measure, so it means something

1. **Report the median of the last 5 checkpoints over 3 seeds.** The best checkpoint of a crowding
   method is mostly legalizer luck.
2. **Report `repair_mean_displacement_cells` beside every number.** It is how much of the answer
   the legalizer wrote.
3. **Run random jitter + legalize on every chip you report.** A gain inside its band isn't one.
4. **Compare against `swap.py`, alone and after your method.**

## Honest limits

- **Three designs, 128 macros, wirelength only.** None of the benchmarks carries a `.lib` or
  `.sdc`, so there is no timing.
- **Adam is not DREAMPlace.** A faithful analytical placer is the real bar for Q3.
- **No critic.** Gradients are cut every `--horizon` steps with nothing to bootstrap the future.

### A core bug this turned up, now fixed

`placax/extras/rewards.py: smoothed_wirelength` produced **NaN gradients** with a correct forward
value: its per-net mask was applied *after* `exp()`, so padded pin slots could overflow float32
(`exp(170.9)` on adaptec1), and reverse-mode AD turned each discarded `0 * inf` into NaN for 23 of
128 macros. Any analytic-gradient method on a padded netlist was silently broken, including the
shipped `shac` agent. The fix masks inside the exponent and guards the empty-net `log(0)`;
regression tests are in `tests/test_rewards.py`. It is the only core change this experiment made.
