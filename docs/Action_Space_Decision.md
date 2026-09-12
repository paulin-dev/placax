# The action space decision, and what it costs

Status: **done. D, then C, then the density term, then SHAC - all of it, in that order, and the
order was the measurement's.** A smoothed wirelength gave essentially complete gradient coverage
for about one percent of fidelity (D, below), which took "drop SHAC" off the table. The
`ActionSpace` protocol followed (C), and now has four implementations rather than the two it was
designed against. The density term - the piece this document called "the one that is genuinely
hard" - is `placax/extras/density.py`, and it was measured the same way before being trusted (see
"What the density term actually buys"). `ContinuousPlacement` and `agents/shac.py` are the last
two, and neither was hard once the three pieces beneath them existed, which is what the ordering
was for.

What remains is not a blocker but a result nobody has: **SHAC has never been run against PPO at a
matched budget on a real design.** The machinery answers the question; it does not answer it.

## Why it is on the table

Three of the eight components in the target architecture are not swappable, and they are not three
independent problems. The action space, the placement representation and the macro ordering are
one decision, taken once, in a single line of `placax/core.py`:

```python
positions = state.positions.at[state.step].set(action)
```

That line fixes: one macro per step, in a pre-computed order, with the action being an integer
`(x, y)` grid cell, into a fixed `(n_macros, 2)` integer array. Everything downstream inherits it -
the `(grid_x, grid_y)` logits map, `sample_action`, `action_log_prob`, the occupancy and boundary
masks, `render`, the `.pl` writer, the whole visualization layer.

None of this is wrong. It is a faithful implementation of the sequential-constructive paradigm
that MaskPlace, EXPlace and ChiPFormer all share. The problem is that it is expressed as though it
were the only paradigm, in a project whose stated purpose is to be the shared environment several
paradigms can be compared in.

## What it currently forecloses, and what it does not

Worth separating, because the audit's framing was too broad and the `Agent` seam has since
settled part of it:

**Not blocked.** Population and sampling methods over whole placements work fine today. The
`RandomSearchAgent` shipped in `placax_agents/agents/baselines.py` is a population method that
never touches the kernel's assumptions, and a GA over per-macro cell choices, or an ACO whose
pheromone table is indexed by `(macro, cell)`, would work the same way. The kernel genuinely does
not care where the action came from - that part of the design claim holds, and is now tested.

**Blocked.** Two things:

1. **Local-search methods** - simulated annealing, most real GAs, FlowPlace's legalizer - need a
   *perturbation* action ("move macro 7 to cell (12, 30)", "swap macros 3 and 9"), not a
   constructive one. `step()` can only append at `state.step`; it has no way to express a change
   to an already-placed macro. This is a small extension, not a redesign.

2. **SHAC**, the comparison the project exists to run, needs `∂reward/∂action`.

## What SHAC actually needs, measured

The environment is differentiable with respect to *positions* - `hpwl()` is a normal JAX
function - and this is what led to the "differentiable environment" framing. Whether that is
enough is an empirical question, so it was measured rather than assumed. On adaptec1, 543 macros:

| quantity | value |
|---|---|
| macros total | 543 |
| macros with at least one net pin | 514 |
| macros receiving a nonzero `∂(-HPWL)/∂position` | **123** |
| connected macros receiving *zero* gradient | **391 (76%)** |
| `∂(occupancy)/∂position` | identically zero |

Two separate problems, and the second is worse than the first.

### What the smoothed surrogate actually buys, measured

Option D said: add the smoothed wirelength as an ordinary `RewardFn` and measure whether the
gradient becomes informative, before rewriting anything. Done. On the same design, as the
fraction of the 514 connected macros receiving a nonzero `d(-objective)/d(position)`:

| objective | gradient density | fidelity vs true HPWL |
|---|---|---|
| raw `-HPWL` | 42.2% (217/514) | exact |
| `-WAWL`, gamma = 0.006 cells | 1.4% | +0.00% |
| `-WAWL`, gamma = 0.17 cells | 57.2% | +0.08% |
| `-WAWL`, gamma = 0.55 cells | 98.8% | +0.33% |
| **`-WAWL`, gamma = 1 grid cell** | **100.0%** | **+1.00%** |
| `-WAWL`, gamma = 2.8 cells | 100.0% | +4.96% |

**One grid cell of smoothing buys complete gradient coverage for one percent of fidelity.** That
is a clear answer: the sparse-gradient half of the problem is solved and cheap.

The measurement also found a bug it is worth not repeating. `gamma` must be expressed **in grid
cells**, not in design units. Log-sum-exp replaces each `max` with `gamma * log(sum(exp(x /
gamma)))`, so gamma has to be commensurate with the coordinates; a real design's coordinates run
to ~1e4, and the registry's original absolute `gamma=1.0` made `exp(x / gamma)` saturate in
float32. The surrogate collapsed back to a hard max and delivered gradient to **1.4%** of
macros - materially worse than raw HPWL, while looking like it was working. The registered
`smoothed` reward now takes `gamma_cells` and scales by `cell_size`.

**What this did not settle, and what settled it.** Density was still zero-gradient and the action
space still discrete. The order of work that followed was: differentiable density term, then the
`ActionSpace` protocol (option C), then SHAC - and all three are now done. The next section is the
density term's own measurement, run before it was trusted for the same reason this one was.

### What the density term actually buys, measured

`extras/density.py` charges the area by which macros over-fill a bin, plus the area they put
outside the canvas. Measured on adaptec1 - 543 macros whose footprints cover 47.7% of the canvas -
as the share of macros receiving a nonzero `d(cost)/d(position)` from a uniformly random placement,
which is what a fresh continuous policy produces:

| `target_density` | overflow (bins) | gradient coverage |
|---|---|---|
| 1.0 | 3804 | 57.3% |
| 0.9 | 5362 | 63.2% |
| 0.8 | 6996 | 70.7% |
| **0.7** | **8713** | **78.6%** |
| 0.5 | 12376 | 67.4% |
| 0.3 | 16352 | 53.4% |

Two things worth having in writing.

**Coverage is not monotonic, and the reason is conservation.** Density is area, and area is
conserved: once *every* bin is above the target, moving a macro shifts area between bins charged
at the same rate and the total overflow does not change. The gradient lives on the frontier
between over-target and under-target bins, so a target below the design's own average density
starts erasing the very thing it was lowered to create. `target_density` is therefore a real knob
with an interior optimum, not a "lower is stricter" dial.

**The same trap as the gamma bug, in a different place.** Measured against the greedy-wiremask
placement at `target_density=1.0`, the term reports 0.0 overflow - correct, that placement is
legal - and *99.8% gradient coverage*, which is nonsense. The greedy placement packs macros on
integer cells, so its bins sit at exactly 1.000, exactly on the hinge of `clip(density - target,
0, None)` - where JAX returns the tie subgradient, 0.5. A number that looks like complete coverage
is an artifact of a placement sitting precisely on a kink. Read coverage beside the cost: where
the cost is zero, coverage means nothing.

### What running it found, which measuring it did not

Two things only a real SHAC run surfaced, both now fixed and both tested:

**A policy will escape the canvas if leaving is cheaper than packing.** With out-of-bounds charged
as squared overhang distance alone, stepping one cell over the edge cost ~1 while overlapping a
2x4-cell macro cost ~8 in overflow. On a 50%-full toy every density weight tried drove overlap to
0% and out-of-bounds to 100% - a perfectly legal placement of an empty canvas. The fix is to charge
the escaped AREA, in the same bin units overflow is charged in, so escaping costs exactly what
overlapping costs; the distance term stays, to keep a gradient on a macro that is already fully
outside and whose escaped area has stopped growing.

**Bounding the corner is not bounding the macro.** The continuous policy's head emits a coordinate
through a sigmoid, which keeps the *corner* on the canvas and says nothing about the body. Saturate
it and the macro sits entirely past the far edge. The head bounds to `canvas - footprint` now,
which is the continuous counterpart of `boundary_mask`: the policy's mean cannot ask for a
placement that does not fit, and the penalty is left to handle the exploration noise.

**HPWL's gradient is sparse by construction.** Half-perimeter wirelength is a sum of per-net
`max - min` over pin coordinates. Only the pins actually *on* a net's bounding box receive
gradient; every pin strictly inside contributes nothing. Three quarters of the connected macros on
this design are interior to every net they belong to, so an analytic policy gradient would move a
quarter of the design and leave the rest untouched. This is a known problem with a known fix -
DREAMPlace does not optimize raw HPWL either, it optimizes a smoothed surrogate (log-sum-exp, or
weighted-average wirelength). Adopting one is a contained change to `placax/extras/rewards.py`
and would make the reward useful to any gradient-based method.

**Legality provides no gradient at all.** `render`, `occupancy_mask` and `boundary_mask` are built
from comparisons, so overlap avoidance is exactly zero-gradient. Today that does not matter,
because legality is enforced by *masking* the action distribution - placements are legal by
construction, which is a genuinely good property of the current design. But a continuous-action
variant cannot mask a continuous distribution the same way, so it needs overlap expressed as a
differentiable penalty. This is the substantial piece of work, and again DREAMPlace has the
recipe: an electrostatic density term.

So the honest statement is not "SHAC needs a new action space". It is: **SHAC needs a continuous
action, a smoothed wirelength, and a differentiable density term, and the environment currently
has none of the three.** The first is cheap, the second is contained, the third is real work.

## Options

**A. Stay discrete; drop SHAC.** Add a perturbation action for local search and leave the
analytic-gradient line of work out. Cost: the project's headline research question goes away, and
"differentiable" should be dropped from the README and the spec, because what is differentiable
today is the *metric*, not the *policy path*.

**B. A second, continuous kernel.** Leave `core.py` alone and add a parallel continuous-placement
kernel with its own relaxed reward, hard-legalized at episode end. Cost: two kernels to keep in
step, which is the exact failure mode the one-kernel design was chosen to avoid - but it is
honest about the fact that the two paradigms really are different, and it never destabilizes the
working discrete path.

**C. Generalize behind an `ActionSpace` protocol.** `step()` delegates "apply this action to this
state" to a swappable object, with `DiscreteGridPlacement` (today's behavior, unchanged),
`ContinuousPlacement`, and `Perturbation` as implementations. Cost: a real refactor of the kernel
and everything reading `positions` directly, and the risk of an abstraction that fits the two
cases we can see and not the third.

**D. Do the reward work first, decide the action space after.** Add the smoothed wirelength and
the differentiable density term as ordinary alternative `RewardFn`s - which the reward axis
already supports, with no kernel change at all - and measure whether the gradient is informative
enough to be worth building an action space around. The measurement above says a quarter of macros
get gradient; the same measurement after smoothing is the number that should decide this.

## Recommendation

**D, then C.** D is now done, and it came out positive, so C stands.

D cost nothing structurally, used the one axis that was already genuinely swappable, and produced
the evidence the rest of the decision needed. The test it had to pass was: does a smoothed
wirelength give dense, well-scaled gradients on a real design? It does - 100% coverage at 1%
fidelity - so option A ("drop SHAC, it was never going to work here") is off the table.

C is therefore the right shape, and there are now **four** concrete action spaces to design the
protocol against rather than two: today's discrete-constructive one, a continuous one for SHAC, a
perturbation one for local search - and legalization, which is a perturbation operator in all but
name. `placax_agents/experiment/registry.py`'s `row_snap` legalizer runs *outside* the episode,
at export time, precisely because the kernel has no action that can move an already-placed macro.
That is a fourth consumer of the same generalization, and it arrived on its own.

What should *not* happen is B by accident - a continuous path bolted on beside the discrete one
because it was faster than generalizing, ending with two kernels that drift apart. That is the
per-paper one-off environment problem reproduced inside a project built to solve it. It did not:
`ContinuousPlacement` is an `ActionSpace` like the other three, driving the same `step()`.

## What to fix in the documents either way

Both of the corrections this section originally asked for have been made, and the situation they
described has changed underneath them:

  * the kernel now runs **six** structurally different agents (PPO, SHAC, greedy wiremask, random
    search, genetic, local search), tested in `tests/test_agents.py`,
    `tests/test_environment_parity.py` and `tests/test_shac.py`;
  * SHAC-versus-PPO really is enabled by the environment's differentiability now, but the sentence
    still has to be precise about which differentiability: `hpwl()` has always had a gradient with
    respect to positions, and what was missing - and now exists - is a smoothed objective every
    macro feels, a legality term that is a cost rather than a mask, and an action the policy can
    be differentiated through.

The remaining honest caveat belongs in the README rather than here: a continuous placement is not
legal by construction the way a masked one is, so a SHAC row that is not 100% legal has not
produced a result, whatever its wirelength says.
