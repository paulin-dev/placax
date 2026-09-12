"""The differentiable density term - the piece SHAC has been blocked on since the first audit.

`docs/Action_Space_Decision.md` ends with a measured order of work: smoothed wirelength (done, and
it bought 100% gradient coverage for 1% fidelity), then **a differentiable density term**, then a
continuous action space, then SHAC. This is the second of those, and the document calls it "the
substantial piece of work" for a specific reason:

    Legality provides no gradient at all. `render`, `occupancy_mask` and `boundary_mask` are built
    from comparisons, so overlap avoidance is exactly zero-gradient.

Legality being a *mask* is a genuinely good property of the discrete path - a placement there is
legal by construction. It is also exactly what a continuous-action method cannot use, because you
cannot mask a continuous distribution the same way. So overlap has to become a differentiable
*cost* instead, which is what this module is.

**What this implements, and what it deliberately does not.** The decision record names
DREAMPlace's electrostatic (ePlace) density as the recipe. That formulation treats density as
charge, solves a Poisson equation per iteration and reads forces off the potential field, which
buys long-range repulsion between macros that do not yet overlap. What is implemented here is the
simpler member of the same family: **exact area overlap between each macro and each bin**, summed
into a density map, penalized above a target. The reasons for starting here rather than there:

  * it is *exactly* differentiable with no solver - the overlap of a rectangle with a bin is a
    product of two clamped differences, so the gradient is analytic and cheap;
  * it needs no DCT/FFT machinery, so it is readable and testable against hand-computed areas;
  * it answers the question that actually blocks SHAC - "is there a useful gradient away from
    overlap at all" - and that question is measurable, which is how every other decision in this
    project has been taken.

**What the gradient actually does, which is not what "overlap penalty" suggests.** This is a
congestion gradient: it moves a macro's area out of over-target bins and into under-target ones.
It is *not* a pairwise repulsion between macros, and the difference shows up in a case worth
knowing about - where the free space lies on the far side of a neighbour, this term pushes a macro
TOWARD that neighbour, because that is the direction that relieves the congested bin. That is
correct for a density formulation and is what DREAMPlace's own family of terms does; it is
recorded here because the intuition "it pushes macros apart" is close enough to be misleading.

The same conservation explains a measured surprise: driving `target_density` very low buys LESS
gradient, not more. Density is conserved area, so once every bin is above the target, moving a
macro shifts area between bins charged at the same rate and the total does not change. The
gradient lives on the frontier between over- and under-target bins, so a target below the design's
own average density starts erasing that frontier. Measured on adaptec1 (47.7% average): coverage
peaks at 78.6% for `target_density=0.7` and falls to 53.4% at 0.3.

Its other known limit is stated rather than discovered later: **a penalty is silent when it is not
violated.** Below the target density the overflow is zero and so is its gradient, so this term
pushes macros apart once they collide and does nothing to spread a merely-crowded region. That is
the right semantics for a legality penalty and the wrong one for a global spreading force, and it
is the difference electrostatics would make. If a measured SHAC run turns out to need spreading
rather than separation, the upgrade is a different `density` implementation behind this same
interface, not a change anywhere else.

**Units.** Like `extras/congestion.py`, positions here are REAL-unit macro centers - what a
RewardFn has after `to_real_centers` - and the density map is in grid bins, so a bin's value is
the fraction of its area covered by macros. That makes `target_density=1.0` mean "a bin may be
completely full and no more", and makes the number comparable across designs of different physical
size.
"""
from placax import _device  # noqa: F401  must run before any `import jax` below
from placax.types import EnvParams

import jax
import jax.numpy as jnp


def _bin_overlaps(lo: jax.Array, hi: jax.Array, n_bins: int) -> jax.Array:
    """`(n_macros, n_bins)` overlap length between each macro's extent and each bin, in bin units.

    One axis of the density computation. `lo`/`hi` are the macro's extent along that axis in GRID
    units, so bin `j` spans `[j, j + 1)` and the overlap is the length of the intersection:

        overlap(i, j) = clamp(min(hi_i, j + 1) - max(lo_i, j), 0, 1)

    Two clamped differences, which is where the gradient comes from: moving a macro changes this
    length continuously on the bins its EDGES fall in, and not at all on the bins it fully covers.
    That is the correct derivative of area with respect to position, and it is why the boundary
    bins carry the signal.
    """
    edges = jnp.arange(n_bins, dtype=jnp.float32)
    overlap = jnp.minimum(hi[:, None], edges[None, :] + 1.0) - jnp.maximum(lo[:, None], edges[None, :])
    return jnp.clip(overlap, 0.0, 1.0)


def area_density(
    positions: jax.Array,
    sizes: jax.Array,
    params: EnvParams,
    cell_size: float,
    placed_mask: jax.Array | None = None,
) -> jax.Array:
    """`(grid_x, grid_y)` map of how much of each bin is covered by macros.

    `positions` are REAL-unit macro CENTERS and `sizes` REAL-unit footprints - pass
    `orientation.effective_sizes(...)` where a run has turns, so a macro on its side occupies the
    area it actually occupies.

    A value of 1.0 means the bin is exactly full. Values above 1.0 mean macros overlap there,
    which is what `density_overflow` charges for.
    """
    if placed_mask is None:
        placed_mask = jnp.ones(positions.shape[0], dtype=bool)
    half = sizes / 2.0
    lo = (positions - half) / cell_size
    hi = (positions + half) / cell_size

    # A rectangle's overlap with a bin is separable into its two axes, so the whole map is one
    # matmul rather than a loop over macros - the same trick rudy_density uses for net boxes.
    over_x = _bin_overlaps(lo[:, 0], hi[:, 0], params.grid_x)
    over_y = _bin_overlaps(lo[:, 1], hi[:, 1], params.effective_grid_y)
    weighted = over_x * placed_mask[:, None].astype(over_x.dtype)
    return weighted.T @ over_y


def density_overflow(density: jax.Array, target_density: float = 1.0) -> jax.Array:
    """Total coverage above `target_density`, summed over bins - the part that cannot physically fit.

    Overflow rather than peak or mean, for the same reason `congestion_overflow` chooses it: peak
    is one bin of noise and mean says nothing about whether anything is infeasible.

    **This is zero, with zero gradient, for any placement that respects the target.** A penalty is
    silent when it is not violated - see this module's docstring.
    """
    return jnp.clip(density - target_density, 0.0, None).sum()


def out_of_bounds_cost(
    positions: jax.Array,
    sizes: jax.Array,
    params: EnvParams,
    cell_size: float,
    placed_mask: jax.Array | None = None,
    distance_weight: float = 1.0,
) -> jax.Array:
    """What macros put outside the canvas costs: the AREA that left, plus how far it went.

    The other half of what masking used to provide for free. `boundary_mask` simply forbade the
    cells a macro would overhang; a continuous action has no such cells to forbid, so leaving the
    canvas has to cost something instead.

    **The area term is what makes this commensurable with overflow, and it was added after a run
    found the alternative.** With the cost measured only as squared overhang distance, a
    continuous policy on a crowded canvas discovered that LEAVING was cheaper than overlapping:
    overlapping a 2x4-cell macro costs ~8 bins of overflow, while stepping one cell over the edge
    cost 1. Measured on a 50%-full toy design, every density weight tried drove overlap to 0% and
    out-of-bounds to 100% - a perfectly legal placement of an empty canvas. Charging the area that
    left, in the same bin units overflow is charged in, makes escaping cost exactly what
    overlapping costs, so neither is an arbitrage on the other.

    The squared-distance term stays, at `distance_weight`, for the case the area term cannot
    handle: once a macro is ENTIRELY outside, its escaped area stops growing and the gradient
    would vanish, leaving it stranded. Distance keeps pulling.
    """
    if placed_mask is None:
        placed_mask = jnp.ones(positions.shape[0], dtype=bool)
    half = sizes / 2.0
    lo = (positions - half) / cell_size
    hi = (positions + half) / cell_size
    canvas = jnp.array([params.grid_x, params.effective_grid_y], dtype=jnp.float32)

    # Area of the macro that is NOT on the canvas, in bin units - directly comparable to overflow.
    extent = jnp.clip(hi - lo, 0.0, None)
    visible = jnp.clip(jnp.minimum(hi, canvas) - jnp.maximum(lo, 0.0), 0.0, None)
    escaped_area = extent[:, 0] * extent[:, 1] - visible[:, 0] * visible[:, 1]

    under = jnp.clip(-lo, 0.0, None)          # how far past the low edge, per axis
    over = jnp.clip(hi - canvas, 0.0, None)   # how far past the high edge, per axis
    distance = (under ** 2 + over ** 2).sum(axis=-1)

    per_macro = escaped_area + distance_weight * distance
    return jnp.where(placed_mask, per_macro, 0.0).sum()


def make_density_cost(
    sizes: jax.Array,
    params: EnvParams,
    cell_size: float,
    target_density: float = 1.0,
    bounds_weight: float = 1.0,
):
    """`(positions, placed_mask) -> scalar` legality cost, differentiable in the positions.

    The drop-in replacement for masking, in the one shape a reward can use: overflow above the
    density target, plus what sticks out of the canvas. Both terms are in grid units, so
    `bounds_weight` trades them against each other on a scale that does not depend on the design's
    physical size.

    Shaped like `congestion.make_congestion_cost` on purpose - a reward that already knows how to
    add one cost to wirelength can add this one the same way.
    """

    def cost(positions: jax.Array, placed_mask: jax.Array | None = None) -> jax.Array:
        density = area_density(positions, sizes, params, cell_size, placed_mask)
        overflow = density_overflow(density, target_density)
        bounds = out_of_bounds_cost(positions, sizes, params, cell_size, placed_mask)
        return overflow + bounds_weight * bounds

    return cost


def gradient_coverage(
    cost_fn, positions: jax.Array, placed_mask: jax.Array | None = None
) -> tuple[int, int]:
    """`(macros receiving a nonzero gradient, macros considered)` - the measurement, not the cost.

    `docs/Action_Space_Decision.md` settled the wirelength half of SHAC's problem with exactly this
    number, and the density half has to be settled the same way rather than assumed: a term that
    is differentiable in principle and delivers gradient to nothing is not progress. Kept here
    beside the cost it measures so the two cannot drift.
    """
    gradient = jax.grad(lambda xy: cost_fn(xy, placed_mask))(positions)
    receiving = (jnp.abs(gradient).sum(axis=-1) > 0)
    if placed_mask is not None:
        receiving = receiving & placed_mask
        considered = int(placed_mask.sum())
    else:
        considered = positions.shape[0]
    return int(receiving.sum()), considered


jitted_area_density = jax.jit(area_density, static_argnames=("params", "cell_size"))

__all__ = [
    "area_density", "density_overflow", "gradient_coverage", "jitted_area_density",
    "make_density_cost", "out_of_bounds_cost",
]
