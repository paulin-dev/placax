"""Composable legality constraints - each function answers one independent question about which cells are illegal."""
from placax import _device  # noqa: F401  must run before any `import jax` below

import jax
import jax.numpy as jnp

from placax.types import EnvParams


def compute_occupied(positions: jax.Array, grid: int) -> jax.Array:
    """Returns a (grid, grid) bool array marking cells that hold a macro's reference point."""
    # 1. Filter out unplaced (-1 sentinel) macros so a JAX scatter/compare doesn't treat -1 as a real index.
    is_placed = positions[:, 0] >= 0
    # 2. Compare every grid cell against every macro's position at once (no Python loop).
    row_idx = jnp.arange(grid)[:, None, None]
    col_idx = jnp.arange(grid)[None, :, None]
    matches = is_placed & (row_idx == positions[:, 0]) & (col_idx == positions[:, 1])
    # 3. A cell counts as occupied if any macro sits there.
    return matches.any(axis=-1)


def occupancy_mask(occupied: jax.Array, macro_size: tuple[int, int]) -> jax.Array:
    """True at (x, y) if placing a macro_size macro there (lower-left corner) overlaps an occupied cell."""
    grid_w, grid_h = occupied.shape
    w, h = macro_size

    # 1. Build a 2D prefix-sum table so any rectangular window sums in O(1).
    cumsum = jnp.pad(occupied.astype(jnp.int32), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)

    # 2. For each cell, work out the macro-sized window starting there, clamped to the grid.
    xs = jnp.arange(grid_w)
    ys = jnp.arange(grid_h)
    x_hi = jnp.clip(xs + w, 0, grid_w)
    y_hi = jnp.clip(ys + h, 0, grid_h)

    # 3. Inclusion-exclusion on the prefix-sum table gives each window's occupied-cell count at once.
    box_sum = (
        cumsum[x_hi][:, y_hi] - cumsum[xs][:, y_hi] - cumsum[x_hi][:, ys] + cumsum[xs][:, ys]
    )
    # 4. Any overlap at all makes that placement illegal.
    return box_sum > 0


def boundary_mask(params: EnvParams, macro_size: tuple[int, int]) -> jax.Array:
    """True at (x, y) if placing a macro_size macro there (lower-left corner) would extend past the canvas edge."""
    w, h = macro_size
    x_illegal = jnp.arange(params.grid_x) + w > params.grid_x
    y_illegal = jnp.arange(params.effective_grid_y) + h > params.effective_grid_y
    return x_illegal[:, None] | y_illegal[None, :]


def quality_mask(scores: jax.Array, max_score: jax.Array) -> jax.Array:
    """True (illegal) wherever scores exceeds max_score; composable with | over any per-cell cost array."""
    return scores > max_score


def lookahead_illegal_masks(occupied: jax.Array, params: EnvParams, macro_sizes: jax.Array) -> jax.Array:
    """Returns occupancy_mask | boundary_mask for each row of macro_sizes, all against today's canvas."""

    # Today's canvas is valid for every lookahead macro, so vmap it over all their sizes.
    def mask_for(size: jax.Array) -> jax.Array:
        w, h = size[0], size[1]
        return occupancy_mask(occupied, (w, h)) | boundary_mask(params, (w, h))

    return jax.vmap(mask_for)(macro_sizes)


def _regularity_axis_cost(axis_len: int, macro_extent: jax.Array, cell_size: float) -> jax.Array:
    """Per-coordinate distance-to-nearest-legal-edge cost along one axis, in real units."""
    # EXPlace's window: only [1, end] carries a cost, so the canvas edge itself is free.
    # `end` is where a macro of this extent last fits, minus the same 1-cell inset.
    end = axis_len - macro_extent - 1
    coords = jnp.arange(axis_len)
    inside = (coords >= 1) & (coords <= end)
    # Distance to the near edge vs. the far edge; the smaller one is what this cell costs.
    to_low = coords
    to_high = end - coords + 1
    return cell_size * jnp.where(inside, jnp.minimum(to_low, to_high), 0)


def regularity_mask(
    params: EnvParams, macro_size: jax.Array, cell_size: float = 1.0, mode: str = "corner"
) -> jax.Array:
    """(grid_x, grid_y) cost map penalizing cells far from the canvas edge - EXPlace's periphery prior.

    mode="corner" sums the two axis costs (pushing macros into corners); mode="edge" takes their
    min (any edge will do). The map is separable, so regularity_cost() reads a single cell in O(1)
    without ever materializing this array - use that on the reward path and this one for observations.
    """
    x_cost = _regularity_axis_cost(params.grid_x, macro_size[0], cell_size)
    y_cost = _regularity_axis_cost(params.effective_grid_y, macro_size[1], cell_size)
    if mode == "corner":
        return x_cost[:, None] + y_cost[None, :]
    if mode == "edge":
        return jnp.minimum(x_cost[:, None], y_cost[None, :])
    raise ValueError(f"unknown regularity mode {mode!r}, expected 'corner' or 'edge'")


def regularity_cost(
    params: EnvParams, macro_size: jax.Array, position: jax.Array, cell_size: float = 1.0, mode: str = "corner"
) -> jax.Array:
    """regularity_mask(...)[position] as a scalar, computed directly from the separable axis terms."""
    x_cost = _regularity_axis_cost(params.grid_x, macro_size[0], cell_size)[position[0]]
    y_cost = _regularity_axis_cost(params.effective_grid_y, macro_size[1], cell_size)[position[1]]
    if mode == "corner":
        return x_cost + y_cost
    if mode == "edge":
        return jnp.minimum(x_cost, y_cost)
    raise ValueError(f"unknown regularity mode {mode!r}, expected 'corner' or 'edge'")


def regularity_max(
    params: EnvParams, macro_size: jax.Array, cell_size: float = 1.0, mode: str = "corner"
) -> jax.Array:
    """Closed-form max of regularity_mask(...), for normalizing the cost to [0, 1] without building the map.

    min(d_low, d_high) peaks where the two meet, at (end + 1) // 2; a macro too big to leave any
    interior (end < 1) has a flat all-zero map, so its max is 0 and callers must guard the divide.
    """

    def axis_max(axis_len: int, macro_extent: jax.Array) -> jax.Array:
        end = axis_len - macro_extent - 1
        return cell_size * jnp.where(end >= 1, (end + 1) // 2, 0)

    x_max = axis_max(params.grid_x, macro_size[0])
    y_max = axis_max(params.effective_grid_y, macro_size[1])
    # Same combining rule as the map itself: a sum's max is the sum of maxes, and a
    # pointwise min's max is the smaller max, since both axis costs hit their peak independently.
    return x_max + y_max if mode == "corner" else jnp.minimum(x_max, y_max)


def lookahead_regularity_masks(
    params: EnvParams, macro_sizes: jax.Array, cell_size: float = 1.0, mode: str = "corner"
) -> jax.Array:
    """Returns regularity_mask() for each row of macro_sizes, mirroring lookahead_illegal_masks()."""
    return jax.vmap(lambda size: regularity_mask(params, size, cell_size, mode))(macro_sizes)
