# The action space decision, and what it costs

Status: **open**. This is a research decision, not an implementation task, so it is written up
rather than taken. It is the one remaining item from the architecture audit that changes the
kernel.

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

**D, then C.**

D costs nothing structurally, uses the one axis that is already genuinely swappable, and produces
the evidence the rest of the decision needs. If a smoothed wirelength plus a density term does not
give dense, well-scaled gradients on a real design, then SHAC was never going to work here and
option A is the honest answer - which is worth finding out before rewriting the kernel, not after.

If it does, C is the right shape, because by then there will be three concrete action spaces to
design the protocol against rather than two, and the perturbation case from local search will have
made the same generalization necessary anyway.

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
