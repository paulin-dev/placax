"""Converts real physical macro sizes (hundreds to thousands of units)
into the small discrete action grid - nothing else in placax does this
conversion; core.py's grid is a pure action-space concept, unaware of
real units at all."""
import jax
import jax.numpy as jnp


def compute_grid_scale(
    sizes_array: jax.Array, grid_x: int, grid_y: int | None = None, target_utilization: float = 0.5
) -> float:
    """Real units per grid cell, sized so all macros together would fill
    roughly target_utilization of the grid's area if packed perfectly.
    grid_y defaults to grid_x (a square grid) - a single cell_size still
    applies uniformly to both dimensions even when they differ, since a
    cell should represent the same real-world size in x and y."""
    if grid_y is None:
        grid_y = grid_x
    total_area = (sizes_array[:, 0] * sizes_array[:, 1]).sum()
    cell_area = total_area / (grid_x * grid_y * target_utilization)
    return float(jnp.sqrt(cell_area))


def to_grid_units(real_size: jax.Array, cell_size: float) -> jax.Array:
    """Real (width, height) -> grid cells, rounded up, minimum 1x1.
    Always a JAX array, not Python ints - needed inside jit/scan, where
    each macro's size is a traced, per-iteration value."""
    return jnp.maximum(1, jnp.ceil(real_size / cell_size).astype(jnp.int32))


def to_real_centers(positions: jax.Array, sizes_array: jax.Array, cell_size: float) -> jax.Array:
    """Grid-cell positions (lower-left corner convention) -> real-unit
    macro centers, matching the units padded_pin_offset is already in -
    the exact formula MaskPlace's own comp_res.py uses (confirmed
    directly): grid_position * ratio + size/2 + offset. Without this
    conversion, hpwl() adds a tiny grid index (0 to grid-1) directly to
    a real-unit offset (hundreds) - a genuine, confirmed bug found this
    session: the offset completely dominates, making the reward barely
    sensitive to the actual grid cell being chosen."""
    return positions.astype(jnp.float32) * cell_size + sizes_array / 2.0
