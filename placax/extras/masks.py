"""Optional, composable legality constraints. Each function answers one
independent question about which grid cells are currently illegal, for
its own specific reason - step() never requires or checks any of these,
and nothing here is stored in EnvState (occupancy is fully derivable
from positions, so computing it on demand - only when someone actually
wants a mask - avoids charging every step() call for a feature most
callers never touch).

compute_occupied() marks only each macro's single reference cell, not
its real footprint - fine if every macro is genuinely 1x1, wrong for
real, differently-sized macros (a real overlap bug was found this way:
a large macro's own interior was left completely unmasked). For real
sizes, build the occupancy source from placax.render.render() instead,
which does account for each macro's true footprint:
    illegal = occupancy_mask(render(positions, sizes, grid), (w, h)) \\
              | boundary_mask(params, (w, h))
Anyone not calling any of these gets identical behavior to before this
module existed."""
from placax import _device  # noqa: F401  must run before any `import jax` below

import jax
import jax.numpy as jnp

from placax.types import EnvParams


def compute_occupied(positions: jax.Array, grid: int) -> jax.Array:
    """Derives the (grid_x, grid_y) occupancy array from placed macro
    positions. Unplaced macros use the (-1, -1) sentinel and must be
    filtered explicitly: JAX's scatter mode='drop' does NOT drop negative
    indices, they wrap like ordinary indexing (-1 means "last row/col")."""
    is_placed = positions[:, 0] >= 0
    row_idx = jnp.arange(grid)[:, None, None]
    col_idx = jnp.arange(grid)[None, :, None]
    xs, ys = positions[:, 0], positions[:, 1]
    matches = is_placed & (row_idx == xs) & (col_idx == ys)
    return matches.any(axis=-1)


def occupancy_mask(occupied: jax.Array, macro_size: tuple[int, int]) -> jax.Array:
    """True at (x, y) if placing a macro_size macro with its lower-left
    corner there would overlap an already-occupied cell. Windows that
    would extend past the edge are clamped to fit, not flagged True -
    that's boundary_mask's job, not this function's; combine with OR.

    Uses a summed-area table (cumulative sum), not dynamic_slice: slice
    windows must have a static size known at trace time, but macro_size
    varies per macro - incompatible with a single scanned rollout over
    many differently-sized macros. A cumulative-sum lookup is ordinary
    indexing, which stays valid even when macro_size is a traced value."""
    grid_w, grid_h = occupied.shape
    w, h = macro_size

    occ_int = occupied.astype(jnp.int32)
    cumsum = jnp.pad(occ_int, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)

    xs = jnp.arange(grid_w)
    ys = jnp.arange(grid_h)
    x_hi = jnp.clip(xs + w, 0, grid_w)
    y_hi = jnp.clip(ys + h, 0, grid_h)

    box_sum = (
        cumsum[x_hi][:, y_hi] - cumsum[xs][:, y_hi] - cumsum[x_hi][:, ys] + cumsum[xs][:, ys]
    )
    return box_sum > 0


def boundary_mask(params: EnvParams, macro_size: tuple[int, int]) -> jax.Array:
    """True at (x, y) if a macro_size macro placed there, lower-left
    corner at (x, y), would extend past the canvas edge. Supports a
    non-square canvas (params.grid_x != params.effective_grid_y)."""
    w, h = macro_size
    xs = jnp.arange(params.grid_x)
    ys = jnp.arange(params.effective_grid_y)
    x_illegal = xs + w > params.grid_x
    y_illegal = ys + h > params.effective_grid_y
    return x_illegal[:, None] | y_illegal[None, :]
