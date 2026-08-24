"""Renders placed macros as filled rectangles on a (grid_x, grid_y)
canvas - an input channel for image-based (CNN) policies. Optional:
step() never calls this."""
from placax import _device  # noqa: F401  must run before any `import jax` below

import jax
import jax.numpy as jnp


def render(positions: jax.Array, sizes: jax.Array, grid_x: int, grid_y: int | None = None) -> jax.Array:
    """(grid_x, grid_y) bool canvas: True wherever any placed macro's
    real footprint covers the cell. positions: (n_macros, 2) grid coords
    with (-1, -1) for unplaced; sizes: (n_macros, 2) (width, height) in
    grid units; grid_y defaults to grid_x."""
    if grid_y is None:
        grid_y = grid_x
    xs = jnp.arange(grid_x)
    ys = jnp.arange(grid_y)

    def footprint_of(pos: jax.Array, size: jax.Array) -> jax.Array:
        x, y = pos
        w, h = size
        in_x = (xs >= x) & (xs < x + w)
        in_y = (ys >= y) & (ys < y + h)
        return (in_x[:, None] & in_y[None, :]) & (x >= 0)

    footprints = jax.vmap(footprint_of)(positions, sizes)
    return footprints.any(axis=0)
