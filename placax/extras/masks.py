"""Composable legality constraints - each function answers one
independent question about which cells are illegal."""
from placax import _device  # noqa: F401  must run before any `import jax` below

import jax
import jax.numpy as jnp

from placax.types import EnvParams


def compute_occupied(positions: jax.Array, grid: int) -> jax.Array:
    """Returns a (grid, grid) bool array marking cells that hold a macro's reference point."""
    # 1. Unplaced macros use the -1 sentinel; filter them out explicitly so a JAX
    #    scatter/compare doesn't treat -1 as a real (wrapped) grid index.
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

    # 1. Build a 2D prefix-sum table so we can sum any rectangular window in O(1),
    #    instead of re-summing cells for every candidate position.
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

    # Today's canvas is valid for every lookahead macro since none of them are
    # placed yet, so we can vmap the same occupied grid over all their sizes.
    def mask_for(size: jax.Array) -> jax.Array:
        w, h = size[0], size[1]
        return occupancy_mask(occupied, (w, h)) | boundary_mask(params, (w, h))

    return jax.vmap(mask_for)(macro_sizes)
