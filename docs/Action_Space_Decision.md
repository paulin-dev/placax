# The action space decision, and what it costs

Status: **option D is done; the decision is now taken on evidence rather than deferred.** The
measurement D asked for has been run (see "What the smoothed surrogate actually buys" below), and
it comes out in favour of continuing: a smoothed wirelength gives essentially complete gradient
coverage for about one percent of fidelity. So the sparse-gradient objection to SHAC is answered,
and what remains blocking it is the differentiable density term and the action space itself - in
that order, and both still unbuilt.

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

**What this does not settle.** Density is still zero-gradient (below), and the action space is
still discrete. The order of work that follows from the measurement is: differentiable density
term, then the `ActionSpace` protocol (option C), then SHAC. The first of those is the one that
is genuinely hard, and it has not become easier.

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
per-paper one-off environment problem reproduced inside a project built to solve it.

What should *not* happen is B by accident - a continuous path bolted on beside the discrete one
because it was faster than generalizing, ending with two kernels that drift apart. That is the
per-paper one-off environment problem reproduced inside a project built to solve it.

## What to fix in the documents either way

`docs/JAX_Placement_Environment_Spec.md` §1 says the kernel "was tested end-to-end with four
structurally different agents". Three exist now (PPO, greedy wiremask, random search) and are
tested in `tests/test_agents.py`; the claim should be corrected to what is true rather than left
in the past tense. §1 also presents SHAC-versus-PPO as enabled by the environment's
differentiability. Per the measurement above that is not the case today, and the sentence should
say what is actually differentiable - the metric with respect to positions - and what that is
missing before an analytic policy gradient can use it.
