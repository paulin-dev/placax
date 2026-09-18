# multiagent — every macro moves itself

A one-week experiment, kept deliberately outside `placax/` and `placax_agents/`. It **imports** the
core (netlist loading, smoothed wirelength, differentiable density cost, greedy warm start,
legality measurement, `experiment.run.score_placement`) and adds one thing the core cannot express:
an action that moves **every macro at once**, one small displacement each, chosen by **one shared
policy applied per macro**.

The question:

> Can a shared per-macro policy, trained by differentiating the placement objective through the
> episode, improve a good placement — and do it better than optimizing the positions directly?

## Quick start

The warm start every method begins from, and the number every method has to beat:

```bash
python -m multiagent.adam --benchmark_dir=benchmarks/adaptec1 --steps=600
```

Train the shared policy (this is the method under test):

```bash
python -m multiagent.train --benchmark_dir=benchmarks/adaptec1 --view=m1 --iterations=200
```

Apply a trained policy to a design it never saw (the claim Adam cannot match):

```bash
python -m multiagent.transfer --run=multiagent/runs/adaptec1-m1-s0 --benchmark_dir=benchmarks/bigblue1
```

Watch it: the policy's episode as a GIF (every macro moving at once, then the repair), the
placement at each checkpoint as training goes, before/after, and curves against Adam:

```bash
python -m multiagent.visualize --run=multiagent/runs/adaptec1-m1-s0 --against multiagent/runs/adaptec1-adam-s0
```

Put the finished runs in one table:

```bash
python -m multiagent.compare multiagent/runs/*
```

Smoke tests (seconds, 8 macros on a 32 grid — run them after every change):

```bash
python -m pytest multiagent/test_multiagent.py -q
```

## What is already measured

All on **adaptec1, 128 macros, 224 grid, core canvas**, one seed, starting from the greedy-wiremask
placement, whose own HPWL is **441,257** (legal, 0% overlap). `raw` is the continuous placement as
the optimizer left it; `repaired` is after `legalize.py` (round, then move each overlapping macro to
the free spot among its 256 nearest that adds the least wire) — the only number a tool would accept.

| method | moves to best | raw HPWL | raw overlap | repaired HPWL | vs greedy |
|---|---|---|---|---|---|
| greedy_wiremask (warm start) | — | 441,257 | 0.00% | 441,257 | — |
| **policy m1**, weight 1, lr 1e-3 | 2,240 (70 × 32) | 203,128 | 30.6% | **334,247** | **+24.3%** |
| adam, weight 1, lr 0.05 | 320 (of 2,560) | 291,254 | 11.8% | 348,433 | +21.0% |
| adam, weight 5 | 600 | 427,287 | 0.8% | 425,770 | +3.5% |

**This is the day-7 go signal, with caveats that belong in the write-up:**

1. **The policy beats Adam at a matched budget** (+24.3% vs +21.0%). Adam was given 2,560 moves
   and stalls by ~960 at `wl_norm` 0.531 (cost flat to 4 decimals after that); the policy keeps
   going to 0.445. A plausible reading is that the policy's sampled, non-stationary rollouts escape
   the local optimum deterministic gradient descent settles in. Both were evaluated 8 times, so
   best-checkpoint selection favours neither.
2. **The repair is doing real work, and more of it for the policy.** The policy hands over a far
   more compressed placement (30.6% overlap) and the repair moves macros ~13–15 cells on average,
   against ~10 for Adam. Every method goes through the identical repair, so the comparison is fair —
   but the honest description is "the policy learns placements that legalize well", not "the
   policy produces legal placements". Report `repair_mean_displacement_cells` beside every number.
3. **One seed, one design, untuned.** Seeds 1–2, the m0/m2 ablation and bigblue1 are what turn this
   into a result.

What the earlier (nearest-spot repair) runs established, and still holds:

- **The gradient path works end to end** — 128 macros each get their own derivative of the shared
  objective, and the policy's wirelength falls steadily from the first iteration.
- **The legality penalty alone cannot produce legal placements.** Weight 1 → −48% raw wirelength at
  16% overlap; weight 25 → −1.5% at 0.5%. A ramp 0.5 → 30 did worse than a constant weight.
- **The repair was the bottleneck.** Switching it from nearest-free-spot to wirelength-aware took
  the same Adam output from +11.1% to +20.0% (see `legalize.py`'s docstring for the sweep).

### A core bug this turned up, now fixed

`placax/extras/rewards.py: smoothed_wirelength` produced **NaN gradients** — with a perfectly
correct forward value. Its per-net mask was applied *after* `exp()`, so padded pin slots (which
read macro 0's coordinates) could overflow float32 inside the exponent: on this design the soft-min
branch reached `exp(170.9)` and gave `+inf` in 21,304 of 23,800 slots, and reverse-mode AD turns
each discarded `0 * inf` into NaN — for 23 of the 128 macros. Any analytic-gradient method on a
padded netlist was silently broken, including the shipped `shac` agent. The fix masks inside the
exponent and guards the empty-net `log(0)`; three regression tests are in `tests/test_rewards.py`.
This is the only core change made from here.

## The pieces

| file | what it is |
|---|---|
| `neighbors.py` | who is wired to whom (clique weights, `2/m` per net), and each macro's top-k partners |
| `context.py` | one loaded design: greedy warm start, bounds, footprints, neighbour table |
| `objective.py` | the shared score — normalized smoothed wirelength + normalized legality cost, with `ramp` |
| `moves.py` | the transition: all macros move at once, clipped to the canvas, gradient cut every `--horizon` |
| `view.py` | what one macro sees: **m0** itself · **m1** + wired neighbours · **m2** + global summary |
| `policy.py` | one shared network, applied per macro row; bounded `max_step * tanh` displacement |
| `train.py` | short-horizon analytic-gradient training (SHAC's window, no critic) |
| `adam.py` | the same objective optimized directly on the positions — the baseline that matters |
| `legalize.py` | round, then move each overlapping macro to the nearby free spot that adds the least wire |
| `transfer.py` | run a trained policy on another design, with no retraining |
| `compare.py` | the runs as one table, with an environment-mismatch warning |
| `visualize.py` | `episode.gif`, `training.gif`, `before_after.png`, `curves.png` into `<run>/viz/` |

Every run directory holds `manifest.json` (written before training, so a crash is still
attributable), `log.jsonl` (one line per iteration), `summary.json`, `best_positions.npy` (the
repaired, legal placement), `best_positions_raw.npy`, and `best_params.pkl`.

## What to tune first, in order

1. **Seeds.** Everything above is one seed. Before tuning anything, run `--seed=1` and `--seed=2`
   for the policy — if +24% does not hold, nothing below matters.
2. **`--density_weight`** (1.0 constant is best so far) and `--max_step`. Lower weights may let the
   policy compress further and lean even harder on the repair — watch the displacement column.
3. **`legalize.py`** still has headroom: macro ordering (largest-first today) and letting a macro
   push a smaller neighbour are the obvious next steps.
4. **`--lr` and `--iterations`** for the policy. It is trained for far fewer moves than Adam gets;
   200 iterations × 32 steps is 6,400 moves, which is the compute-matched Adam row (`--steps=6400`).
5. **`--horizon`**, the one hyperparameter the method is actually about. If longer windows help, a
   value-function bootstrap (real SHAC) is the next step.

## The week

- **Days 1–2** — Tune the repair and the weight until the policy beats its own warm start reliably.
  Run `m0` / `m1` / `m2` at 2–3 seeds each. If `m0` fails and `m1` works, that is a finding.
- **Days 3–4** — Adam at matched move budgets (`--steps` = iterations × steps) on adaptec1 and
  bigblue1, 3 seeds. Then `transfer.py`: adaptec1 → bigblue1, against Adam from scratch on
  bigblue1. This is the headline experiment.
- **Day 5** — Scale: `--macro_budget=0` (every macro, 543 on adaptec1) versus 128, for both
  methods. Centralized comparison: `scripts/compare_agents.py --agents=greedy_wiremask,shac,ppo`
  at a matched budget, so the table has a one-macro-per-step row.
- **Days 6–7** — DREAMPlace + OpenROAD on each method's `best_positions.npy`, then write up.
  Leave the last day for writing, not running.

## Honest limits, for the write-up

- **No critic**, so this is SHAC's window without its bootstrap. The log says `short_horizon`
  rather than `shac` for that reason.
- **Legality is a penalty plus a repair**, not a guarantee. Every number is reported both ways.
- **One design tested so far** (adaptec1). bigblue1 works through the same code path; ariane133 has
  no placement rows, so it needs `--canvas=die`.
- **No timing anywhere.** None of the benchmarks here carries a `.lib` or an `.sdc`, so WNS/TNS
  cannot be computed — do not put them in the plan.
- **The `m2` global summary is the only channel between macros.** There is no communication
  protocol, deliberately.
