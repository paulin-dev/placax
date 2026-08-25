"""Composable legality constraints - each function answers one
independent question about which cells are illegal."""
from placax import _device  # noqa: F401  must run before any `import jax` below

import jax
import jax.numpy as jnp

from placax.types import EnvParams


def compute_occupied(positions: jax.Array, grid: int) -> jax.Array:
    """(grid, grid) bool array of occupied reference cells only (not real
    footprints - use render()+occupancy_mask() for that). Unplaced
    macros (-1 sentinel) are filtered explicitly, else JAX scatter would
    wrap negative indices instead of dropping them."""
    is_placed = positions[:, 0] >= 0
    # Broadcast-compare every grid cell against every macro's position at once.
    row_idx = jnp.arange(grid)[:, None, None]
    col_idx = jnp.arange(grid)[None, :, None]
    matches = is_placed & (row_idx == positions[:, 0]) & (col_idx == positions[:, 1])
    return matches.any(axis=-1)  # cell is occupied if any macro sits there


def occupancy_mask(occupied: jax.Array, macro_size: tuple[int, int]) -> jax.Array:
    """True at (x, y) if a macro_size macro placed there (lower-left
    corner) overlaps an occupied cell. Edge cases are boundary_mask's
    job - combine with OR. Summed-area table so macro_size can be traced."""
    grid_w, grid_h = occupied.shape
    w, h = macro_size

    # 2D prefix-sum table: cumsum[i, j] = occupied cells in [0,i) x [0,j).
    cumsum = jnp.pad(occupied.astype(jnp.int32), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)

    xs = jnp.arange(grid_w)
    ys = jnp.arange(grid_h)
    x_hi = jnp.clip(xs + w, 0, grid_w)  # each cell's macro-sized window, clamped to the grid
    y_hi = jnp.clip(ys + h, 0, grid_h)

    # Inclusion-exclusion: sum over any window in O(1) from the prefix-sum table.
    box_sum = (
        cumsum[x_hi][:, y_hi] - cumsum[xs][:, y_hi] - cumsum[x_hi][:, ys] + cumsum[xs][:, ys]
    )
    return box_sum > 0


def boundary_mask(params: EnvParams, macro_size: tuple[int, int]) -> jax.Array:
    """True at (x, y) if a macro_size macro placed there (lower-left
    corner) would extend past the canvas edge."""
    w, h = macro_size
    x_illegal = jnp.arange(params.grid_x) + w > params.grid_x
    y_illegal = jnp.arange(params.effective_grid_y) + h > params.effective_grid_y
    return x_illegal[:, None] | y_illegal[None, :]


def quality_mask(scores: jax.Array, max_score: jax.Array) -> jax.Array:
    """True (illegal) wherever scores exceeds max_score - a single
    composable cutoff over any per-cell cost array (e.g.
    extras.rewards.wiremask(), a congestion or density map, ...).
    Combine with |  like any other mask here. How max_score is picked is
    up to the caller: scores.min() + margin (an additive cutoff) and
    jnp.quantile(scores, keep_frac) (a relative cutoff) are both just
    ordinary uses of this one primitive, not special cases of it."""
    return scores > max_score


def lookahead_illegal_masks(occupied: jax.Array, params: EnvParams, macro_sizes: jax.Array) -> jax.Array:
    """(horizon, grid_x, grid_y) bool: occupancy_mask | boundary_mask for
    each row of macro_sizes (horizon, 2) against the SAME occupied canvas
    - valid because none of those macros are placed yet, so they all mask
    against today's canvas. Pairs with
    policy.observation.lookahead_sizes()'s grid-unit equivalent."""

    def mask_for(size: jax.Array) -> jax.Array:
        w, h = size[0], size[1]
        return occupancy_mask(occupied, (w, h)) | boundary_mask(params, (w, h))

    return jax.vmap(mask_for)(macro_sizes)
