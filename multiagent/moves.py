"""The transition: every macro moves by its own small step, all of them in the same step.

This is the one genuinely new thing in this directory. The core's four action spaces all move
exactly one macro per step - three constructive ones append to `state.step`, and `Perturbation`
writes to an index the action names. A decentralized formulation cannot be expressed as either:
the action is `(n_macros, 2)`, one displacement per macro, applied at once.

**Why the step is bounded and small.** A macro that can teleport anywhere makes the starting
placement irrelevant and turns refinement back into construction. Bounded nudges (`max_step`, in
grid cells) keep every intermediate placement close to a legal one, which is what lets the density
penalty do its job with a weight that is not tuned to the edge of stability. It is also the one
defence against the oscillation every multi-agent placement formulation is warned about: two
macros chasing each other can only chase at `max_step` per step.

**Why clipping, not a penalty, for the canvas.** `hi = canvas - footprint` bounds the macro's
BODY, so a clipped placement is entirely on the canvas by construction - the continuous
counterpart of `boundary_mask`. `clip` has a perfectly good gradient (one inside, zero outside),
and the out-of-bounds half of the density cost stays in the objective anyway to keep a signal on
exploration noise that pushes outward.

**Where the gradient flows, and where it is cut.** A displacement at step `t` changes the
placement, and every later step's cost reads that placement - so a macro's action keeps receiving
gradient from the future of the episode. Every `horizon` steps the carried placement is passed
through `stop_gradient`, which truncates that path: this is SHAC's short window, written exactly
the way `placax_agents/agents/shac.py` writes it (`jnp.where(is_boundary, stop_gradient(leaf),
leaf)`), so the two are comparable line for line.
"""
from typing import Callable

from placax import _device  # noqa: F401  must precede jax imports

import jax
import jax.numpy as jnp


def apply_deltas(
    positions: jax.Array, deltas: jax.Array, lo: jax.Array, hi: jax.Array
) -> jax.Array:
    """`clip(positions + deltas)` - the whole transition. Differentiable in both arguments.

    The result keeps `positions`' dtype. That is not cosmetic: this project runs with x64 enabled,
    so a single float64 leaking out of a policy parameter would silently widen the placement, and
    `lax.scan` rejects a carry whose dtype changed - as a type error several steps away from the
    line that caused it.
    """
    return jnp.clip(positions + deltas, lo, hi).astype(positions.dtype)


def rollout(
    positions: jax.Array,
    key: jax.Array,
    act: Callable[[jax.Array, dict, jax.Array, jax.Array], jax.Array],
    objective,
    lo: jax.Array,
    hi: jax.Array,
    steps: int,
    horizon: int,
    density_weight=None,
) -> tuple[jax.Array, jax.Array]:
    """Runs one episode of simultaneous moves, returning `(final_positions, costs)`.

    `act(positions, parts, progress, key) -> deltas` is the decision rule - a policy in `train.py`,
    and nothing at all in the tests, which drive this with a fixed displacement to check the
    transition on its own.

    `costs` has `steps + 1` entries: the cost of the placement BEFORE each move, plus the cost of
    the final placement. Every entry is on the same scale as every other, so the mean over them is
    a sensible loss (it rewards getting low early and staying there) and `costs[0]` is always the
    warm start - the number the episode has to beat.
    """
    # Which steps end a window. A static array rather than arithmetic on the scan index, matching
    # the shipped SHAC agent; the last step always closes one so nothing dangles.
    boundaries = ((jnp.arange(steps) + 1) % horizon == 0).at[steps - 1].set(True)
    progress = jnp.arange(steps, dtype=jnp.float32) / max(steps, 1)

    def scan_step(carried, xs):
        step_key, is_boundary, step_progress = xs
        # 1. Score where we are. The same numbers feed the loss and (under m2) the observation, so
        #    the policy sees exactly what it is being judged on, at no extra cost.
        parts = objective.parts(carried, density_weight)
        # 2. Every macro decides its own displacement from its own view of this placement.
        deltas = act(carried, parts, step_progress, step_key)
        moved = apply_deltas(carried, deltas, lo, hi)
        # 3. The window boundary: cut the path backwards from here.
        moved = jnp.where(is_boundary, jax.lax.stop_gradient(moved), moved)
        return moved, parts["total"]

    final, costs = jax.lax.scan(
        scan_step, positions, (jax.random.split(key, steps), boundaries, progress)
    )
    return final, jnp.concatenate([costs, objective.total(final, density_weight)[None]])
