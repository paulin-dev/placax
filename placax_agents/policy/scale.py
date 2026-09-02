"""Converts real physical macro sizes to/from the discrete action grid."""
import jax
import jax.numpy as jnp


def compute_grid_scale(
    sizes_array: jax.Array, grid_x: int, grid_y: int | None = None, target_utilization: float = 0.5
) -> float:
    """Real units per grid cell, sized so all macros fill roughly target_utilization of the grid area."""
    if grid_y is None:
        grid_y = grid_x
    total_area = (sizes_array[:, 0] * sizes_array[:, 1]).sum()
    return float(jnp.sqrt(total_area / (grid_x * grid_y * target_utilization)))


def to_grid_units(real_size: jax.Array, cell_size: float) -> jax.Array:
    """Real (width, height) -> grid cells, rounded up, minimum 1x1 (stays a JAX array for jit/scan)."""
    return jnp.maximum(1, jnp.ceil(real_size / cell_size).astype(jnp.int32))


def to_real_centers(positions: jax.Array, sizes_array: jax.Array, cell_size: float) -> jax.Array:
    """Grid positions (lower-left corner) -> real-unit macro centers."""
    return positions.astype(jnp.float32) * cell_size + sizes_array / 2.0


def to_real_lower_left(positions: jax.Array, cell_size: float) -> jax.Array:
    """Grid positions (lower-left corner) -> real-unit lower-left corner, e.g. for Bookshelf .pl / DEF PLACED."""
    return positions.astype(jnp.float32) * cell_size
