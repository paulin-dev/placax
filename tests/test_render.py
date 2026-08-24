from placax.extras.render import render  # noqa: F401  must precede jax imports

import jax.numpy as jnp


def test_render_marks_correct_footprint_per_macro_size() -> None:
    positions = jnp.array([[1, 1], [-1, -1], [5, 5]])
    sizes = jnp.array([[2, 3], [4, 4], [1, 1]])
    canvas = render(positions, sizes, grid_x=8)

    assert canvas[1:3, 1:4].all()  # macro 0's real 2x3 footprint
    assert canvas.sum() == 2 * 3 + 1  # macro 0's footprint + macro 2's single cell
    assert canvas[5, 5]
    assert not canvas[3, 3]  # not covered by anything


def test_render_all_unplaced_is_empty_canvas() -> None:
    positions = jnp.full((4, 2), -1)
    sizes = jnp.ones((4, 2), dtype=jnp.int32)
    canvas = render(positions, sizes, grid_x=4)
    assert not canvas.any()


def test_render_overlapping_footprints_still_correct() -> None:
    # two macros whose footprints overlap - canvas should just show the union
    positions = jnp.array([[0, 0], [1, 1]])
    sizes = jnp.array([[2, 2], [2, 2]])
    canvas = render(positions, sizes, grid_x=4)
    assert canvas[0, 0] and canvas[1, 1] and canvas[2, 2]
    assert canvas.sum() == 7  # union of two overlapping 2x2 blocks


def test_render_rectangular_canvas() -> None:
    # Regression test: an earlier version only accepted a single square
    # grid dimension - real chip die areas aren't always square.
    positions = jnp.array([[1, 2]])
    sizes = jnp.array([[2.0, 3.0]])
    canvas = render(positions, sizes, grid_x=4, grid_y=6)
    assert canvas.shape == (4, 6)
    assert int(canvas.sum()) == 6  # x in [1,3), y in [2,5) -> 2*3 cells
