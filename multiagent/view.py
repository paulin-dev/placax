"""What ONE macro sees. This is the experiment, not a detail.

Three levels, each a strict superset of the one before, so a difference between two runs is a
difference in information and nothing else:

    m0  itself only         where it is, how big it is, how far it is from each edge
    m1  + its neighbours    the k macros it is most strongly wired to, relative to itself
    m2  + global numbers    how the whole placement is currently doing

The reason to build the ablation before tuning anything: if `m0` cannot improve a placement and
`m1` can, that is the finding - wirelength is a property of pairs, so a macro that cannot see its
partners has no gradient information about which direction helps, and no amount of training fixes
it. If `m0` works, the local-rule ("swarm") reading of the problem is much stronger than expected.
Either way the comparison answers something; a single tuned configuration answers nothing.

**Everything is normalized to the canvas** - positions and sizes as a fraction of the grid,
distances as a fraction of the grid, neighbour weights into [0, 1] by `neighbors.top_k_neighbors`.
A policy trained on adaptec1 has to produce sensible actions on bigblue1, whose canvas is a
different size in every respect, and raw grid coordinates would make the two designs look like
different problems to the same network.

**Neighbour positions are RELATIVE** (`neighbour - me`), for the same reason: "my partner is 40
cells to my left" transfers between designs and "my partner is at x=93" does not.
"""
from typing import Callable

from placax import _device  # noqa: F401  must precede jax imports

import jax
import jax.numpy as jnp

from multiagent.context import Context

LEVELS = ("m0", "m1", "m2")
"""The observability ladder, in the order the experiment walks it."""


def make(ctx: Context, level: str = "m1") -> tuple[Callable, int]:
    """Returns `(view_fn, n_features)`.

    `view_fn(positions, parts, progress) -> (n_macros, n_features)`, where `parts` is the
    objective's own dict for this placement and `progress` is how far the episode has run. Both
    are ignored below `m2`; they are still taken, so the rollout calls every level identically.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown view level {level!r}; choose one of {', '.join(LEVELS)}")

    canvas = ctx.canvas
    sizes = ctx.sizes_grid / canvas                      # footprint as a fraction of the canvas
    neighbor_idx = ctx.neighbor_idx
    neighbor_weight = ctx.neighbor_weight
    neighbor_valid = ctx.neighbor_valid.astype(jnp.float32)
    k = neighbor_idx.shape[1]

    def local(positions: jax.Array) -> jax.Array:
        """m0: the 8 numbers a macro can know about itself without looking at anything else."""
        xy = positions / canvas
        # Room between this macro's own edges and the canvas's, in both directions. The macro's
        # body is already bounded by moves.apply_deltas, so these are all in [0, 1].
        room_low = xy
        room_high = 1.0 - (xy + sizes)
        return jnp.concatenate([xy, sizes, room_low, room_high], axis=-1)

    def wired(positions: jax.Array) -> jax.Array:
        """m1's addition: k * 6 numbers about the macros this one is actually connected to."""
        xy = positions / canvas
        neighbor_xy = xy[neighbor_idx]                    # (n, k, 2)
        relative = neighbor_xy - xy[:, None, :]           # where they are, from HERE
        neighbor_sizes = sizes[neighbor_idx]              # (n, k, 2)
        extra = jnp.stack([neighbor_weight, neighbor_valid], axis=-1)  # (n, k, 2)
        per_neighbor = jnp.concatenate([relative, neighbor_sizes, extra], axis=-1)
        # Zero out slots that are not real connections, so a macro with two partners does not read
        # a third partner's geometry as if it mattered. The valid flag survives this untouched
        # (0 * 0 = 0, 1 * 1 = 1), so the policy can still tell an empty slot from a partner
        # sitting exactly on top of it.
        per_neighbor = per_neighbor * neighbor_valid[..., None]
        return per_neighbor.reshape(positions.shape[0], k * 6)

    def global_state(positions: jax.Array, parts: dict, progress: jax.Array) -> jax.Array:
        """m2's addition: 3 numbers every macro sees the same copy of.

        Broadcast rather than concatenated per macro for a reason worth writing down: this is the
        only channel through which one macro can notice what the others did, and it is a summary,
        not a message. No communication protocol, which is the thing to avoid in week one.
        """
        shared = jnp.stack([
            parts["wl_norm"] - 1.0,      # 0 at the warm start, negative once wires got shorter
            parts["legal_norm"],         # fraction of macro area overlapping or off-canvas
            progress,                    # how much of the episode is left
        ])
        return jnp.broadcast_to(shared, (positions.shape[0], shared.shape[0]))

    if level == "m0":
        def view_fn(positions, _parts, _progress):
            return local(positions)
        return view_fn, 8

    if level == "m1":
        def view_fn(positions, _parts, _progress):
            return jnp.concatenate([local(positions), wired(positions)], axis=-1)
        return view_fn, 8 + 6 * k

    def view_fn(positions, parts, progress):
        return jnp.concatenate(
            [local(positions), wired(positions), global_state(positions, parts, progress)],
            axis=-1,
        )
    return view_fn, 8 + 6 * k + 3
