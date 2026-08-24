from placax_agents.policy.scale import compute_grid_scale, to_grid_units, to_real_centers  # noqa: F401

import jax.numpy as jnp


def test_compute_grid_scale_hand_example() -> None:
    # 4 macros, each 10x10 (area 100 each, total 400). grid=4 (16 cells),
    # target_utilization=0.5 -> cell_area = 400 / (16*0.5) = 50 -> cell_size = sqrt(50)
    sizes = jnp.array([[10.0, 10.0]] * 4)
    cell_size = compute_grid_scale(sizes, grid_x=4, target_utilization=0.5)
    assert abs(cell_size - 50**0.5) < 1e-4


def test_compute_grid_scale_rectangular_grid() -> None:
    # Same total cell count (8*2=16) as the square case above should
    # give the exact same cell_size - the grid's shape shouldn't matter,
    # only its total area in cells.
    sizes = jnp.array([[10.0, 10.0]] * 4)
    cell_size = compute_grid_scale(sizes, grid_x=8, grid_y=2, target_utilization=0.5)
    assert abs(cell_size - 50**0.5) < 1e-4


def test_to_grid_units_rounds_up_minimum_one() -> None:
    assert to_grid_units(jnp.array([5.0, 5.0]), cell_size=10.0).tolist() == [1, 1]
    assert to_grid_units(jnp.array([25.0, 5.0]), cell_size=10.0).tolist() == [3, 1]
    assert to_grid_units(jnp.array([0.0, 0.0]), cell_size=10.0).tolist() == [1, 1]  # never zero


def test_to_real_centers_matches_comp_res_formula() -> None:
    # Real example: macro at grid (10,20), size (500,2136), ratio=10,
    # verified against MaskPlace's own comp_res.py formula by hand.
    positions = jnp.array([[10, 20]])
    sizes_array = jnp.array([[500.0, 2136.0]])
    centers = to_real_centers(positions, sizes_array, cell_size=10.0)
    assert centers.tolist() == [[350.0, 1268.0]]


def test_to_real_centers_with_real_offset_matches_hand_calculation() -> None:
    positions = jnp.array([[10, 20]])
    sizes_array = jnp.array([[500.0, 2136.0]])
    centers = to_real_centers(positions, sizes_array, cell_size=10.0)
    pin_pos = centers[0] + jnp.array([-248.5, 16.0])
    assert pin_pos.tolist() == [101.5, 1284.0]
