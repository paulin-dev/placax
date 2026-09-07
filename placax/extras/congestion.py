"""RUDY: a routing-congestion proxy, so the reward axis has a second objective to compare against.

The research plan's reward comparison is "HPWL vs. HPWL+congestion vs. a learned predictor, agent
held fixed". Only wirelength existed, so the first of those could not be run from a config at all
and the axis was swappable in principle rather than in practice.

RUDY (Rectangular Uniform wire DensitY) is the standard cheap congestion estimate: every net
spreads its expected wire demand uniformly over its own bounding box. A net whose bounding box is
w x h bins is assumed to need about (w + h) worth of wire inside an area of w*h bins, so it
contributes (w + h) / (w * h) demand to each bin it covers. Bins where the total exceeds capacity
are congested. It is a proxy, not a router - it knows nothing about layers, vias, blockages or
detours - but it correlates well enough with real congestion to be the usual pre-routing signal,
and it costs a matmul.

**Cost, stated plainly.** The naive form is O(nets x grid^2) per evaluation. The implementation
below exploits the fact that a bounding-box indicator is separable - the outer product of an
x-range indicator and a y-range indicator - so the whole density map is two masked matmuls rather
than a loop over nets. That makes it affordable once per episode; it is still far more expensive
than HPWL, which is why the reward factory defaults to sparse.

**Not differentiable in the useful sense.** The bin indicators are comparisons, so the gradient
with respect to macro positions is zero almost everywhere. That is fine for PPO, which treats the
reward as a black box, and is exactly NOT fine for an analytic-gradient method - the same
limitation legality has (see docs/Action_Space_Decision.md). A smooth congestion term is the same
class of work as the differentiable density term SHAC would need, and is not attempted here.
"""
from placax import _device  # noqa: F401  must run before any `import jax` below
from placax.types import EnvParams

import jax
import jax.numpy as jnp

_BIG = jnp.float32(1e9)


def net_bounding_boxes(
    positions: jax.Array,
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    placed_mask: jax.Array | None = None,
):
    """(lo, hi, counted) per net in REAL units - the same bounding boxes hpwl() sums the spans of."""
    if placed_mask is None:
        placed_mask = jnp.ones(positions.shape[0], dtype=bool)
    pin_xy = positions[padded_pin_idx].astype(jnp.float32) + padded_pin_offset
    counted = valid_mask & placed_mask[padded_pin_idx]
    lo = jnp.where(counted[..., None], pin_xy, _BIG).min(axis=1)
    hi = jnp.where(counted[..., None], pin_xy, -_BIG).max(axis=1)
    # A net needs two counted pins before it has a box worth routing through.
    has_box = counted.sum(axis=1) >= 2
    return lo, hi, has_box


def rudy_density(
    positions: jax.Array,
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    params: EnvParams,
    cell_size: float,
    placed_mask: jax.Array | None = None,
) -> jax.Array:
    """Per-bin wire demand over the canvas. `positions` are REAL-unit macro centers.

    Returns a (grid_x, grid_y) array in units of "wire length per bin area", directly comparable
    across designs of different physical size because both the demand and the bins are measured
    in the same grid units.
    """
    lo, hi, has_box = net_bounding_boxes(
        positions, padded_pin_idx, padded_pin_offset, valid_mask, placed_mask
    )
    grid_x, grid_y = params.grid_x, params.effective_grid_y

    # Real units -> grid bins, clamped to the canvas: a pin outside it still routes to the edge.
    lo_bin = jnp.clip(lo / cell_size, 0.0, None)
    hi_bin = jnp.clip(hi / cell_size, None, jnp.array([grid_x, grid_y], dtype=jnp.float32))

    # Box extent in bins, floored at one bin so a degenerate (zero-width) box still has demand.
    width = jnp.maximum(hi_bin[:, 0] - lo_bin[:, 0], 1.0)
    height = jnp.maximum(hi_bin[:, 1] - lo_bin[:, 1], 1.0)
    # RUDY's estimate: (w + h) of wire spread over w*h of area.
    demand = jnp.where(has_box, (width + height) / (width * height), 0.0)

    # A bounding box's indicator is separable, so the density map is an outer product per net -
    # and the sum over nets of those outer products is one matmul, not a loop over nets.
    xs = jnp.arange(grid_x, dtype=jnp.float32)[None, :]      # (1, grid_x)
    ys = jnp.arange(grid_y, dtype=jnp.float32)[None, :]      # (1, grid_y)
    in_x = ((xs + 1.0 > lo_bin[:, 0:1]) & (xs < hi_bin[:, 0:1])).astype(jnp.float32)
    in_y = ((ys + 1.0 > lo_bin[:, 1:2]) & (ys < hi_bin[:, 1:2])).astype(jnp.float32)
    return (in_x * demand[:, None]).T @ in_y                 # (grid_x, grid_y)


def congestion_overflow(density: jax.Array, capacity: float = 1.0) -> jax.Array:
    """Total demand above `capacity`, summed over bins - the part that cannot actually be routed.

    Overflow rather than peak or mean: peak is one bin's worth of noise, mean says nothing about
    whether anything is actually infeasible, and overflow is what a router reports.
    """
    return jnp.clip(density - capacity, 0.0, None).sum()


def peak_congestion(density: jax.Array) -> jax.Array:
    """The worst bin's demand, for reporting alongside overflow."""
    return density.max()


def make_congestion_cost(
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    params: EnvParams,
    cell_size: float,
    capacity: float = 1.0,
):
    """(positions, placed_mask) -> scalar congestion overflow, ready to add to a reward."""

    def cost(positions: jax.Array, placed_mask: jax.Array | None = None) -> jax.Array:
        density = rudy_density(
            positions, padded_pin_idx, padded_pin_offset, valid_mask, params, cell_size,
            placed_mask,
        )
        return congestion_overflow(density, capacity)

    return cost


jitted_rudy_density = jax.jit(rudy_density, static_argnames=("params", "cell_size"))
