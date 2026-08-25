"""Renders placed macros as filled rectangles on a (grid_x, grid_y) canvas."""
from placax import _device  # noqa: F401  must run before any `import jax` below

import jax
import jax.numpy as jnp


def render(positions: jax.Array, sizes: jax.Array, grid_x: int, grid_y: int | None = None) -> jax.Array:
    """(grid_x, grid_y) bool canvas, True where a placed macro's
    footprint covers the cell. positions: -1,-1 sentinel for unplaced."""
    if grid_y is None:
        grid_y = grid_x
    xs = jnp.arange(grid_x)
    ys = jnp.arange(grid_y)

    def footprint_of(pos: jax.Array, size: jax.Array) -> jax.Array:
        # This macro's w x h rectangle as a boolean grid; (x >= 0) drops unplaced macros.
        x, y = pos
        w, h = size
        in_x = (xs >= x) & (xs < x + w)
        in_y = (ys >= y) & (ys < y + h)
        return (in_x[:, None] & in_y[None, :]) & (x >= 0)

    footprints = jax.vmap(footprint_of)(positions, sizes)  # one footprint per macro, batched
    return footprints.any(axis=0)  # flatten to one canvas: covered if any macro is there
