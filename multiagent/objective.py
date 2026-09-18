"""The one score every agent and every baseline here is judged by.

Two terms, both differentiable in the macro positions, because the whole method depends on that:

  * **wirelength**, as the smoothed (log-sum-exp) surrogate rather than raw HPWL. Raw HPWL gives
    zero gradient to 76% of adaptec1's connected macros - only the pins ON a net's bounding box
    feel it - and one grid cell of smoothing buys 100% coverage for 1% fidelity
    (docs/Action_Space_Decision.md). A per-macro policy whose gradient reaches a quarter of the
    macros would be a different experiment than the one intended.

  * **legality**, as `extras/density.py`'s cost: bin overflow above a target, plus the area a
    macro puts off the canvas. Masking is not available to a continuous action, so legality has to
    be a cost. It is a PENALTY, so it is silent when satisfied - which is the right semantics here,
    since the episode starts from a legal placement and the term's job is to keep it that way.

**Both terms are normalized, and the normalizers are recorded.** Wirelength is divided by the warm
start's own wirelength, so `wl_norm = 1.0` means "no better than greedy" and 0.98 means "2% better"
on every design. Legality is divided by the total macro footprint in bins, so `legal_norm` reads as
a fraction of the macro area that is overlapping or off-canvas. Without this, `density_weight`
would have to be re-derived per benchmark from the HPWL magnitude - which is exactly the trap
`scripts/measure_reward_terms.py` exists to warn about.

**Two units deliberately not used here.** `cell_size=None` is passed to the wirelength: the core's
reward quantizes pins to the grid lattice with `jnp.round`, which is correct for scoring a discrete
placement and fatal for a gradient (round has a zero derivative everywhere it is defined). And
`gamma` is given in grid CELLS and multiplied by `cell_size` on the way in, because log-sum-exp
needs gamma commensurate with the coordinates - an absolute `gamma=1.0` on coordinates running to
1e4 saturates float32 and silently degenerates back to a hard max.

`real_hpwl` in every log line is the un-smoothed, un-normalized metric, computed by the runner's
own `score_placement` so it is the same number `scripts/compare_agents.py` prints.
"""
from dataclasses import dataclass
from typing import Callable

from placax import _device  # noqa: F401  must precede jax imports
from placax.extras.density import make_density_cost
from placax.extras.rewards import smoothed_wirelength
from placax_agents.experiment.run import score_placement
from placax_agents.policy.scale import to_real_centers

import jax
import jax.numpy as jnp

from multiagent.context import Context

DEFAULT_GAMMA_CELLS = 1.0
"""Smoothing width, in grid cells: measured at 100% gradient coverage for +1.00% fidelity."""


def ramp(start: float, end: float, fraction: float) -> float:
    """The legality weight at `fraction` through a run, geometrically from `start` to `end`.

    Ramping the penalty is what analytical placers do, and for a reason this project measured on
    its own objective: at a weight low enough to let wirelength improve (1.0), Adam on adaptec1
    reached -48% raw wirelength with 15.9% overlap, and the repair gave back most of it; at a
    weight high enough to hold overlap near zero (25.0), wirelength barely moved (-1.5%). Neither
    fixed weight is the answer - a low one first, to find a better arrangement, then a high one, to
    make it legal, is. Geometric rather than linear because the useful range spans orders of
    magnitude, and `start == end` is exactly no ramp at all.
    """
    if start <= 0.0 or end <= 0.0:
        raise ValueError(f"density weights are multipliers on a cost, got {start} -> {end}")
    return float(start * (end / start) ** min(max(fraction, 0.0), 1.0))


DEFAULT_TARGET_DENSITY = 0.95
"""Deliberately not 1.0. The greedy warm start packs macros onto integer cells, so its bins sit at
exactly 1.000 - precisely on the kink of `clip(density - target, 0, None)`, where JAX returns the
tie subgradient 0.5 and "99.8% gradient coverage" is an artifact of a placement sitting on a hinge
(docs/Action_Space_Decision.md measured exactly this). A target slightly below 1.0 puts the start
just inside the penalized region, so the term has a real, readable value from step 0."""


@dataclass(frozen=True)
class Objective:
    """The score, its pieces, and the normalizers they were divided by."""

    total: Callable[..., jax.Array]
    """`(positions, density_weight=None) -> scalar` in grid units. This gets differentiated."""

    parts: Callable[..., dict]
    """`(positions, density_weight=None) -> {wl, wl_norm, legal, legal_norm, total}`, all traced.

    `density_weight` overrides the run's default for this call, and may be a TRACED scalar - which
    is what lets the weight be ramped during a run without recompiling anything. Passing None uses
    the default the objective was built with."""

    wl0: float
    """The warm start's smoothed wirelength - the wirelength normalizer, in real design units."""

    area_bins: float
    """Total macro footprint in grid bins - the legality normalizer."""

    density_weight: float
    target_density: float
    gamma_cells: float
    overlap_weight: float = 0.0


def make(
    ctx: Context,
    density_weight: float = 1.0,
    target_density: float = DEFAULT_TARGET_DENSITY,
    gamma_cells: float = DEFAULT_GAMMA_CELLS,
    bounds_weight: float = 1.0,
    overlap_weight: float = 0.0,
) -> Objective:
    """Builds the objective for one context. Called once per run; the closures are jit-friendly."""
    benchmark = ctx.benchmark
    cell_size = benchmark.cell_size
    sizes = benchmark.sizes_array

    density_cost = make_density_cost(
        sizes, ctx.params, cell_size, target_density=target_density,
        bounds_weight=bounds_weight,
    )

    def wirelength(positions: jax.Array) -> jax.Array:
        centers = to_real_centers(positions, sizes, cell_size)
        return smoothed_wirelength(
            centers, benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
            None, None, gamma_cells * cell_size,
        )

    def legality_cost(positions: jax.Array) -> jax.Array:
        return density_cost(to_real_centers(positions, sizes, cell_size), None)

    grid_sizes = ctx.sizes_grid
    upper = jnp.triu(jnp.ones((ctx.n_macros, ctx.n_macros), dtype=bool), k=1)

    def overlap_area(positions: jax.Array) -> jax.Array:
        """Total pairwise overlap area, in grid cells squared - what the repair has to undo.

        The density term measures congestion per bin, which two macros can share a little of
        while every bin stays under target; this is the overlap itself. Its gradient pushes each
        overlapping pair apart along both axes, and is zero for a pair that does not touch.
        """
        lo, hi = positions, positions + grid_sizes
        width = jnp.minimum(hi[:, None, 0], hi[None, :, 0]) - jnp.maximum(lo[:, None, 0], lo[None, :, 0])
        height = jnp.minimum(hi[:, None, 1], hi[None, :, 1]) - jnp.maximum(lo[:, None, 1], lo[None, :, 1])
        return jnp.where(upper, jax.nn.relu(width) * jax.nn.relu(height), 0.0).sum()

    # The two normalizers, computed once from the placement the agents start from.
    wl0 = float(wirelength(ctx.warm_start))
    area_bins = float((ctx.sizes_grid[:, 0] * ctx.sizes_grid[:, 1]).sum())

    def parts(positions: jax.Array, weight=None) -> dict:
        wl = wirelength(positions)
        legal = legality_cost(positions)
        wl_norm = wl / wl0
        legal_norm = legal / area_bins
        weight = density_weight if weight is None else weight
        overlap_norm = overlap_area(positions) / area_bins
        return {
            "wl": wl,
            "wl_norm": wl_norm,
            "legal": legal,
            "legal_norm": legal_norm,
            "overlap_norm": overlap_norm,
            "total": wl_norm + weight * legal_norm + overlap_weight * overlap_norm,
        }

    def total(positions: jax.Array, weight=None) -> jax.Array:
        return parts(positions, weight)["total"]

    return Objective(
        total=total, parts=parts, wl0=wl0, area_bins=area_bins,
        density_weight=density_weight, target_density=target_density, gamma_cells=gamma_cells,
        overlap_weight=overlap_weight,
    )


def report(ctx: Context, objective: Objective, positions: jax.Array) -> dict:
    """Plain-Python metrics for one placement: the objective's parts, real HPWL, and legality.

    `real_hpwl` is measured on the float positions (the runner's definition) and
    `real_hpwl_snapped` on the rounded ones - the placement a tool would actually receive. Both,
    because the gap between them is how much of an improvement survives being written to a file,
    and a continuous method that only wins before rounding has not won.
    """
    parts = {name: float(value) for name, value in objective.parts(positions).items()}
    snapped = jnp.round(positions)
    return {
        **parts,
        "real_hpwl": score_placement(ctx.benchmark, positions),
        "real_hpwl_snapped": score_placement(ctx.benchmark, snapped),
        **ctx.legality_of(positions),
    }
