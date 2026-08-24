"""Renders placed macros as filled rectangles on a (grid_x, grid_y)
canvas - the same 'view' concept as MaskPlace's viewmask, useful as an
input channel for an image-based (CNN) policy. Optional, like
everything in masks.py - step() doesn't call this, nobody is required
to."""
from placax import _device  # noqa: F401  must run before any `import jax` below

import jax
import jax.numpy as jnp


def render(positions: jax.Array, sizes: jax.Array, grid_x: int, grid_y: int | None = None) -> jax.Array:
    """positions: (n_macros, 2) grid coords, (-1, -1) = unplaced.
    sizes: (n_macros, 2) (width, height) per macro, in the same grid units.
    grid_y defaults to grid_x (a square canvas) - pass both explicitly
    for a rectangular one.
    Returns a (grid_x, grid_y) bool canvas: True wherever any placed
    macro's real footprint covers that cell."""
    if grid_y is None:
        grid_y = grid_x
    xs = jnp.arange(grid_x)
    ys = jnp.arange(grid_y)

    def footprint_of(pos: jax.Array, size: jax.Array) -> jax.Array:
        x, y = pos
        w, h = size
        placed = x >= 0
        in_x = (xs >= x) & (xs < x + w)
        in_y = (ys >= y) & (ys < y + h)
        return (in_x[:, None] & in_y[None, :]) & placed

    footprints = jax.vmap(footprint_of)(positions, sizes)
    return footprints.any(axis=0)
