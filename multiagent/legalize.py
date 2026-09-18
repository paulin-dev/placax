"""Turning a continuous placement into a legal one: round, then repair what rounding leaves.

**Why this is needed at all, measured rather than assumed.** A continuous method has no mask, so
legality is a penalty, and a penalty buys what it is priced at. Measured with Adam on adaptec1
(128 macros, 224 grid, core canvas, 400 steps), the density weight trades exactly as expected and
never reaches zero overlap on its own:

    density_weight    wirelength vs greedy    overlap (of macro area)
             1.0              -38.7%                    13.25%
             5.0               -3.3%                     0.92%
            25.0               -1.5%                     0.46%

The reason is structural, and `extras/density.py` says so itself: that term is a CONGESTION
measure - area over a per-bin target - not a pairwise overlap. Two macros can share a little area
while every bin stays near its target, and rounding a legal float placement onto the integer grid
can create overlap that was not there before. So no weight makes the raw output legal, and a weight
large enough to try destroys the wirelength gain the method exists to produce.

**The standard answer, which this implements.** Analytical placement has always been two phases:
global placement, which allows overlap and optimizes wirelength, then legalization, which removes
the overlap at the smallest cost it can. DREAMPlace does it, EGPlace's contribution is largely a
better version of it. This is the simple member of that family: snap to the grid, then walk macros
in descending area and move each to its NEAREST free legal position.

**Why wirelength-aware by default, measured.** Nearest-free-spot is the obvious repair, and it is
blind to the one thing being optimized: a macro pushed one cell left can cost far less wire than
one pushed one cell right. So the default looks at the `candidates` nearest free spots and takes
the one whose nets come out shortest. On the same Adam output (adaptec1, 128 macros, raw 298,387
HPWL with overlap), against the greedy warm start's 441,257:

    repair                  repaired HPWL    vs greedy    mean move
    nearest                     461,621        -4.6%        3.7 cells
    wirelength, k=16            458,769        -4.0%        3.5 cells
    wirelength, k=64            371,423       +15.8%        9.4 cells
    wirelength, k=256           352,920       +20.0%       10.2 cells   <- default
    wirelength, k=1024          352,872       +20.0%       10.7 cells

It saturates at 256, and costs 0.2 s either way. The larger mean move is the price, and it is
reported in every log line (`repair_mean_displacement_cells`) so a reader can see how much of the
answer the repair supplied.

**Largest-first** because a big macro has few legal positions and a small one has many - placing
the constrained macros while the canvas is still empty is what keeps the total displacement small.

**Every method gets the same repair.** The greedy warm start is already legal on integer cells, so
repairing it is the identity and its number does not move - which matters more now the repair can
improve wirelength by itself: a macro whose own spot is free is always left there, so the repair
never gets the chance to "improve" a placement that did not need repairing. Adam and the policy are both repaired
before being scored, so the comparison is between placements that a tool would actually accept.
"""
from functools import partial

from placax import _device  # noqa: F401  must precede jax imports
from placax_agents.policy.scale import to_grid_units

import jax
import jax.numpy as jnp
import numpy as np

from multiagent.context import Context


def _free_positions(occupied_integral: np.ndarray, width: int, height: int) -> np.ndarray:
    """`(nx, ny)` boolean: is the `width x height` box with this lower-left corner free?

    From the integral image of the occupancy grid, so the whole candidate map costs four array
    reads rather than a loop over cells - on a 224 grid that is 50,176 candidate positions per
    macro, which is not something to test one at a time.
    """
    grid_x, grid_y = occupied_integral.shape[0] - 1, occupied_integral.shape[1] - 1
    nx, ny = grid_x - width + 1, grid_y - height + 1
    if nx <= 0 or ny <= 0:
        return np.zeros((max(nx, 0), max(ny, 0)), dtype=bool)
    x0 = np.arange(nx)[:, None]
    y0 = np.arange(ny)[None, :]
    covered = (
        occupied_integral[x0 + width, y0 + height]
        - occupied_integral[x0, y0 + height]
        - occupied_integral[x0 + width, y0]
        + occupied_integral[x0, y0]
    )
    return covered == 0


class _Wiring:
    """Per-macro view of the netlist, for scoring candidate positions by the wirelength they add.

    Built once per repair from the padded arrays. For macro `m`, `nets[m]` lists the nets it is on
    and, for each, which pin slots are its own - the ones that move with a candidate position - and
    which belong to other macros, which stay where the repair has put them so far.
    """

    def __init__(self, ctx: Context):
        benchmark = ctx.benchmark
        self.cell_size = float(benchmark.cell_size)
        self.half_sizes = np.asarray(benchmark.sizes_array, dtype=np.float64) / 2.0
        self.pin_idx = np.asarray(benchmark.padded_pin_idx)
        self.pin_offset = np.asarray(benchmark.padded_pin_offset, dtype=np.float64)
        self.valid = np.asarray(benchmark.valid_mask)
        n_macros = ctx.n_macros
        self.nets = [[] for _ in range(n_macros)]
        for net in range(self.pin_idx.shape[0]):
            slots = np.flatnonzero(self.valid[net])
            owners = self.pin_idx[net, slots]
            for macro in np.unique(owners):
                self.nets[macro].append((net, slots[owners == macro], slots[owners != macro]))

    def added_wirelength(self, macro: int, candidates: np.ndarray, reference: np.ndarray):
        """HPWL of `macro`'s nets with it at each of `candidates` - `(n_candidates,)`, real units.

        `reference` holds every other macro's current lower-left grid position. Only the nets this
        macro is on can change, so only they are scored; the rest of the design is a constant and
        drops out of the comparison between candidates.
        """
        centers = reference * self.cell_size + self.half_sizes            # (n_macros, 2) real units
        own_centers = candidates * self.cell_size + self.half_sizes[macro]  # (n_candidates, 2)
        total = np.zeros(len(candidates))
        for net, own_slots, other_slots in self.nets[macro]:
            own_pins = own_centers[:, None, :] + self.pin_offset[net, own_slots][None, :, :]
            lo, hi = own_pins.min(axis=1), own_pins.max(axis=1)
            if other_slots.size:
                other_pins = centers[self.pin_idx[net, other_slots]] + self.pin_offset[net, other_slots]
                lo = np.minimum(lo, other_pins.min(axis=0))
                hi = np.maximum(hi, other_pins.max(axis=0))
            total += (hi - lo).sum(axis=1)
        return total


REPAIR_MODES = ("nearest", "wirelength")


def repair(
    ctx: Context,
    positions: jnp.ndarray,
    mode: str = "wirelength",
    candidates: int = 256,
) -> tuple[np.ndarray, dict]:
    """Rounds `positions` to the grid and moves overlapping macros to legal free spots.

    `mode="nearest"` takes the closest free spot. `mode="wirelength"` looks at the `candidates`
    closest free spots and takes the one whose nets come out shortest, given where every other
    macro currently is. Displacement alone is the wrong thing to minimize: a macro pushed one cell
    left can cost far less wire than one pushed one cell right, and nearest-only cannot tell them
    apart. The candidate count keeps it a REPAIR - the macro still lands near where the optimizer
    wanted it - rather than a second placer that ignores the first one's answer.

    Returns `(legal_positions, stats)`. `stats` records how much the repair had to move things -
    `moved_macros` and `max_displacement_cells` - because a repair that relocates half the design
    has quietly replaced the method's answer with its own, and a result should say so.
    """
    if mode not in REPAIR_MODES:
        raise ValueError(f"unknown repair mode {mode!r}; choose one of {', '.join(REPAIR_MODES)}")
    grid_x, grid_y = ctx.params.grid_x, ctx.params.effective_grid_y
    # Integer footprints, from the same conversion the legality measurement uses - so a placement
    # this function calls legal is legal by `extras/legality.py`'s definition, not by a looser one.
    footprints = np.asarray(to_grid_units(ctx.benchmark.sizes_array, ctx.benchmark.cell_size))
    wanted = np.asarray(jnp.round(positions)).astype(np.int32)
    wiring = _Wiring(ctx) if mode == "wirelength" else None
    # Where every macro is "for now": its wanted spot until the repair reaches it, its final spot
    # after. Unrepaired macros still pull on their neighbours from where the optimizer put them.
    reference = wanted.astype(np.float64)

    occupied = np.zeros((grid_x, grid_y), dtype=np.int32)
    placed = np.zeros_like(wanted)
    displacements = np.zeros(len(wanted), dtype=np.float32)

    # Largest macro first: it has the fewest legal positions, so it should choose while the canvas
    # is still empty.
    areas = footprints[:, 0].astype(np.int64) * footprints[:, 1].astype(np.int64)
    for macro in np.argsort(-areas):
        width, height = int(footprints[macro, 0]), int(footprints[macro, 1])
        integral = np.zeros((grid_x + 1, grid_y + 1), dtype=np.int32)
        integral[1:, 1:] = occupied.cumsum(axis=0).cumsum(axis=1)
        free = _free_positions(integral, width, height)
        target_x = int(np.clip(wanted[macro, 0], 0, max(free.shape[0] - 1, 0)))
        target_y = int(np.clip(wanted[macro, 1], 0, max(free.shape[1] - 1, 0)))
        if free.size == 0 or not free.any():
            # Nothing legal is left for this macro. Keep the optimizer's own choice, clamped, and
            # let the legality measurement report it rather than inventing a position.
            placed[macro] = (target_x, target_y)
            displacements[macro] = np.inf
            continue
        # Free positions ranked by L1 distance on the grid: the repair should be a small correction
        # that makes the placement legal, not a re-placement.
        distance = (
            np.abs(np.arange(free.shape[0])[:, None] - target_x)
            + np.abs(np.arange(free.shape[1])[None, :] - target_y)
        )
        distance = np.where(free, distance, np.iinfo(np.int32).max).ravel()
        if wiring is None or distance[target_x * free.shape[1] + target_y] == 0:
            # Nearest mode - or the macro's own spot is free, in which case leaving it alone is
            # both the smallest move and the optimizer's own choice.
            flat = int(distance.argmin())
        else:
            n_free = int(free.sum())
            shortlist = np.argpartition(distance, min(candidates, n_free) - 1)[:min(candidates, n_free)]
            spots = np.stack(np.divmod(shortlist, free.shape[1]), axis=1).astype(np.float64)
            cost = wiring.added_wirelength(int(macro), spots, reference)
            # Ties (common: a macro on no net scores 0 everywhere) go to the nearer spot.
            flat = int(shortlist[np.lexsort((distance[shortlist], cost))[0]])
        x, y = divmod(flat, free.shape[1])
        placed[macro] = (x, y)
        reference[macro] = (x, y)
        displacements[macro] = abs(x - target_x) + abs(y - target_y)
        occupied[x:x + width, y:y + height] += 1

    finite = displacements[np.isfinite(displacements)]
    stats = {
        "repair_moved_macros": int((displacements > 0).sum()),
        "repair_max_displacement_cells": float(finite.max()) if finite.size else 0.0,
        "repair_mean_displacement_cells": float(finite.mean()) if finite.size else 0.0,
        "repair_unplaceable_macros": int((~np.isfinite(displacements)).sum()),
    }
    return placed, stats


def spread(ctx: Context, positions: jnp.ndarray, steps: int = 300,
           max_move: float = 0.25) -> jnp.ndarray:
    """Pushes overlapping macros apart by small amounts before the repair sees them.

    **Why this exists, measured.** `repair` can only move an overlapping macro into a hole that is
    already free. On a sparse canvas that hole is nearby; on ariane133 - half the canvas covered by
    identical 17 x 11-cell SRAMs - it is not, and one cell of random jitter made the repair move
    macros ~28 cells on average and cost ~10% HPWL. Every refinement method lost on ariane for that
    reason alone, Adam included.

    This is Boids' separation rule as a legalizer: descent on the pairwise overlap area ALONE,
    measured on the integer footprints. Only overlapping pairs have a gradient, so a macro that
    overlaps nothing does not move until a neighbour pushes into it - a packed row makes room a
    little from each neighbour instead of one macro teleporting. No margin is needed: footprints are
    whole cells and rounding is monotone, so two floats a footprint apart stay apart when rounded.

    Two details that each broke it when wrong. Overlap is `where(w > 0, w, 0)`, not `max(w, 0)`:
    JAX gives `max` a 0.5 subgradient at a tie, and the greedy start is full of blocks touching
    edge to edge, which then pushed each other apart. And the step is plain gradient descent capped
    at `max_move` cells per macro, not Adam: Adam rescales a vanishing gradient to a full step.
    """
    footprints = jnp.asarray(to_grid_units(ctx.benchmark.sizes_array, ctx.benchmark.cell_size),
                             dtype=jnp.float32)
    hi = jnp.clip(jnp.asarray(ctx.canvas, dtype=jnp.float32) - footprints, 0.0, None)
    start = jnp.clip(jnp.asarray(positions, dtype=jnp.float32), 0.0, hi)
    return _spread(start, footprints, hi, steps, max_move)


@partial(jax.jit, static_argnames=("steps",))
def _spread(xy, footprints, hi, steps, max_move):
    """`spread`'s loop, compiled once per design shape rather than once per call."""
    n = xy.shape[0]
    upper = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)

    def overlap(xy):
        lo_, hi_ = xy, xy + footprints
        w = jnp.minimum(hi_[:, None, 0], hi_[None, :, 0]) - jnp.maximum(lo_[:, None, 0], lo_[None, :, 0])
        h = jnp.minimum(hi_[:, None, 1], hi_[None, :, 1]) - jnp.maximum(lo_[:, None, 1], lo_[None, :, 1])
        hit = upper & (w > 0) & (h > 0)
        return jnp.where(hit, w * h, 0.0).sum()

    def step(xy, _):
        grads = jax.grad(overlap)(xy)
        norm = jnp.linalg.norm(grads, axis=1, keepdims=True)
        move = grads / jnp.maximum(norm, 1e-9) * jnp.minimum(norm, max_move)
        return jnp.clip(xy - move, 0.0, hi), None

    xy, _ = jax.lax.scan(step, xy, None, length=steps)
    return xy


DEFAULT_SPREAD_STEPS = 300
"""The portfolio is on by default: every entry point legalizes both ways and keeps the better."""


def repair_and_report(ctx: Context, objective, positions: jnp.ndarray,
                      spread_steps: int = DEFAULT_SPREAD_STEPS):
    """`(legal_positions, metrics)` - the repaired placement and its metrics, keys prefixed.

    Both entry points score every placement twice: once as the optimizer left it (`eval_*`, which
    can be illegal) and once after this repair (`eval_repaired_*`, which a tool would accept). The
    second set is the result; the first set is what says whether the repair had to rescue it.

    `spread_steps > 0` legalizes TWO ways - the plain repair, and `spread` followed by the repair -
    and keeps whichever legal result has the lower HPWL, recording which in
    `repair_used_spread`. Neither wins everywhere, measured with random jitter on the warm start:
    spreading cuts ariane133's damage from -9.6% to -2.1% (a packed canvas has no nearby holes),
    and makes bigblue1's worse (on a 3.5%-full canvas the wire-aware choice among 256 free spots
    is itself an optimizer). Every method gets the same portfolio, so it stays a fair judge.
    """
    from multiagent import objective as objective_mod

    wanted = np.asarray(jnp.round(positions)).astype(np.int32)
    options = [(False, positions)]
    if spread_steps > 0:
        options.append((True, spread(ctx, positions, steps=spread_steps)))
    best = None
    for used_spread, candidate in options:
        placed, stats = repair(ctx, candidate)
        # Displacement from the placement the METHOD handed over - the spread is part of the
        # repair, and its moves count against it like any other.
        moved = np.abs(placed - wanted).sum(axis=1)
        stats["repair_moved_macros"] = int((moved > 0).sum())
        stats["repair_max_displacement_cells"] = float(moved.max())
        stats["repair_mean_displacement_cells"] = float(moved.mean())
        stats["repair_used_spread"] = used_spread
        metrics = objective_mod.report(ctx, objective, jnp.asarray(placed, dtype=jnp.float32))
        key = (not metrics["is_legal"], metrics["real_hpwl_snapped"])
        if best is None or key < best[0]:
            best = (key, placed, metrics, stats)
    _, placed, metrics, stats = best
    return placed, {**{f"repaired_{name}": value for name, value in metrics.items()}, **stats}
