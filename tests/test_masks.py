from placax.extras.masks import (  # noqa: F401  both must precede jax imports
    boundary_mask,
    compute_occupied,
    lookahead_illegal_masks,
    lookahead_regularity_masks,
    occupancy_mask,
    quality_mask,
    regularity_cost,
    regularity_mask,
    regularity_max,
)
from placax.types import EnvParams  # noqa: F401

import jax
import jax.numpy as jnp


def test_compute_occupied_marks_placed_positions() -> None:
    positions = jnp.array([[1, 1], [2, 3]])
    occupied = compute_occupied(positions, grid=4)
    assert occupied[1, 1]
    assert occupied[2, 3]
    assert occupied.sum() == 2


def test_compute_occupied_ignores_unplaced_sentinel_not_wraps_it() -> None:
    # Regression test: a naive scatter-based implementation using
    # mode='drop' does NOT drop negative indices the way it sounds like
    # it should - JAX wraps them like ordinary indexing, so (-1, -1)
    # silently became (3, 3) on a 4x4 grid when first tried (confirmed
    # empirically, not assumed). This must never happen.
    positions = jnp.array([[1, 1], [-1, -1], [2, 3]])
    occupied = compute_occupied(positions, grid=4)
    assert not occupied[3, 3]
    assert occupied.sum() == 2  # only the two genuinely placed positions


def test_compute_occupied_all_unplaced_is_empty() -> None:
    positions = jnp.full((5, 2), -1)
    occupied = compute_occupied(positions, grid=4)
    assert not occupied.any()


def test_occupancy_mask_1x1_macro_flags_only_occupied_cell() -> None:
    occupied = jnp.zeros((4, 4), dtype=bool).at[1, 1].set(True)
    mask = occupancy_mask(occupied, macro_size=(1, 1))
    assert mask[1, 1]
    assert mask.sum() == 1


def test_occupancy_mask_expands_by_macro_size_minus_one() -> None:
    # 2x2 occupied block on an 8x8 grid; a 2x2 macro's *starting* position
    # is illegal not just where it exactly overlaps, but up to (size-1)
    # cells before it too, since the footprint extends forward from there.
    occupied = jnp.zeros((8, 8), dtype=bool).at[3:5, 3:5].set(True)
    mask = occupancy_mask(occupied, macro_size=(2, 2))
    assert mask[2:5, 2:5].all()  # 3x3 illegal region, verified by hand
    assert mask.sum() == 9
    assert not mask[0, 0]
    assert not mask[7, 7]


def test_boundary_mask_flags_only_positions_that_would_overflow() -> None:
    params = EnvParams(grid=4)
    mask = boundary_mask(params, macro_size=(2, 2))
    # x+2>4 or y+2>4 -> only index 3 in either axis
    assert mask[3, :].all()
    assert mask[:, 3].all()
    assert not mask[0, 0]
    assert not mask[2, 2]  # 2x2 macro fits exactly: [2,3]x[2,3], within grid


def test_boundary_mask_1x1_macro_never_illegal() -> None:
    params = EnvParams(grid=4)
    mask = boundary_mask(params, macro_size=(1, 1))
    assert not mask.any()


def test_masks_compose_with_or() -> None:
    params = EnvParams(grid=4)
    occupied = jnp.zeros((4, 4), dtype=bool).at[0, 0].set(True)
    combined = occupancy_mask(occupied, (1, 1)) | boundary_mask(params, (1, 1))
    assert combined[0, 0]  # illegal via occupancy
    assert not combined[1, 1]  # legal via either check


def test_occupancy_mask_works_with_traced_macro_size_in_scan() -> None:
    # Regression test: an earlier implementation used dynamic_slice, whose
    # window size must be static at trace time - incompatible with a
    # scanned rollout where each macro has a different size. This must
    # keep working with macro_size as a traced, per-iteration value.
    def scan_body(carry, macro_size):
        occupied = jnp.zeros((8, 8), dtype=bool).at[3:5, 3:5].set(True)
        params = EnvParams(grid=8)
        illegal = occupancy_mask(occupied, macro_size) | boundary_mask(params, macro_size)
        return carry, illegal.sum()

    sizes = jnp.array([[2, 2], [3, 1], [1, 4]])
    _, results = jax.lax.scan(scan_body, None, sizes)
    assert results.tolist() == [24, 24, 34]  # verified by hand for the first case


def test_quality_mask_flags_cells_above_the_cutoff() -> None:
    scores = jnp.array([[0.0, 5.0], [2.0, 10.0]])
    mask = quality_mask(scores, max_score=jnp.array(2.0))
    assert mask.tolist() == [[False, True], [False, True]]


def test_quality_mask_composes_with_legality_masks() -> None:
    params = EnvParams(grid=2)
    occupied = jnp.zeros((2, 2), dtype=bool)
    scores = jnp.array([[0.0, 5.0], [0.0, 0.0]])
    combined = occupancy_mask(occupied, (1, 1)) | boundary_mask(params, (1, 1)) | quality_mask(scores, jnp.array(1.0))
    assert combined.tolist() == [[False, True], [False, False]]


def test_lookahead_illegal_masks_stacks_one_mask_per_macro_size() -> None:
    params = EnvParams(grid=4)
    occupied = jnp.zeros((4, 4), dtype=bool).at[0, 0].set(True)
    macro_sizes = jnp.array([[1, 1], [2, 2]])
    masks = lookahead_illegal_masks(occupied, params, macro_sizes)
    assert masks.shape == (2, 4, 4)
    assert masks[0, 0, 0]  # 1x1 macro: illegal exactly where occupied
    assert masks[1, 0, 0]  # 2x2 macro's footprint also covers (0,0) -> illegal
    assert not masks[0, 3, 3]  # 1x1 fits fine there
    assert masks[1, 3, 3]  # 2x2 macro would overflow the grid there


def test_boundary_mask_rectangular_grid() -> None:
    # Regression test: an earlier version assumed a square grid
    # everywhere - real chip die areas aren't always square.
    params = EnvParams(grid=4, grid_y=6)
    mask = boundary_mask(params, macro_size=(2, 2))
    assert mask.shape == (4, 6)
    assert bool(mask[3, :].all())  # last column, w=2 overflows grid_x=4
    assert bool(mask[:, 5].all())  # last row, h=2 overflows grid_y=6
    assert not bool(mask[0, 0])  # fits fine


def _explace_regularity_reference(grid, size_x, size_y, coef_x, coef_y, mode):
    """Verbatim port of EXPlace's get_regularity_mask (place_env.py:559), for differential testing."""
    import numpy as np

    start_x = start_y = 1
    end_x, end_y = grid - size_x - 1, grid - size_y - 1
    rows, cols = np.arange(grid), np.arange(grid)
    x1 = np.where((rows >= start_x) & (rows <= end_x), rows - start_x + 1, 0)
    x2 = np.where((rows >= start_x) & (rows <= end_x), end_x - rows + 1, 0)
    x_mask = (coef_x * np.minimum(x1, x2))[:, None].repeat(grid, axis=1)
    y1 = np.where((cols >= start_y) & (cols <= end_y), cols - start_y + 1, 0)
    y2 = np.where((cols >= start_y) & (cols <= end_y), end_y - cols + 1, 0)
    y_mask = (coef_y * np.minimum(y1, y2))[None, :].repeat(grid, axis=0)
    return np.minimum(x_mask, y_mask) if mode == "edge" else x_mask + y_mask


def test_regularity_mask_matches_explace_reference() -> None:
    # Differential test against the upstream numpy implementation across a sweep of grids,
    # macro sizes, both modes and a non-unit cell size. Exact agreement, not just close.
    import numpy as np

    for grid in (8, 16, 64, 224):
        for size_x in (1, 3, 7, 15):
            for size_y in (1, 2, 11):
                for mode in ("corner", "edge"):
                    for cell_size in (1.0, 12.5):
                        got = regularity_mask(EnvParams(grid=grid), jnp.array([size_x, size_y]), cell_size, mode)
                        want = _explace_regularity_reference(grid, size_x, size_y, cell_size, cell_size, mode)
                        assert np.allclose(np.asarray(got), want), (grid, size_x, size_y, mode, cell_size)


def test_regularity_corner_mode_frees_only_the_corners_edge_mode_frees_the_whole_border() -> None:
    # This is the real difference between the two modes, and it is easy to get backwards:
    # in corner mode sitting on the left edge still costs you for being far from top/bottom,
    # so only the four corners are actually free.
    size = jnp.array([1, 1])
    corner = regularity_mask(EnvParams(grid=16), size, mode="corner")
    edge = regularity_mask(EnvParams(grid=16), size, mode="edge")

    for x, y in ((0, 0), (0, 15), (15, 0), (15, 15)):
        assert float(corner[x, y]) == 0.0
    assert float(corner[0, 8]) > 0.0  # mid-edge is not free in corner mode
    assert corner[0, :].sum() > 0

    # edge mode takes the min of the two axis costs, so any boundary cell is free.
    assert edge[0, :].sum() == 0 and edge[:, 0].sum() == 0
    assert edge[15, :].sum() == 0 and edge[:, 15].sum() == 0
    assert float(edge[8, 8]) > 0.0


def test_regularity_mask_peaks_in_the_canvas_interior() -> None:
    # min(dist_low, dist_high) peaks where the two meet, so the max sits at the centre.
    mask = regularity_mask(EnvParams(grid=16), jnp.array([1, 1]))
    peak_x, peak_y = (int(a) for a in jnp.unravel_index(jnp.argmax(mask), mask.shape))
    assert peak_x in (7, 8) and peak_y in (7, 8)
    assert float(mask[8, 8]) == float(mask.max())


def test_regularity_edge_mode_is_cheaper_than_corner_mode() -> None:
    # corner sums both axis costs, edge takes their min, so edge can never cost more.
    size = jnp.array([2, 3])
    corner = regularity_mask(EnvParams(grid=32), size, mode="corner")
    edge = regularity_mask(EnvParams(grid=32), size, mode="edge")
    assert (edge <= corner).all()
    assert float(edge.max()) < float(corner.max())


def test_regularity_max_matches_the_maps_actual_max() -> None:
    # regularity_max() is a closed form; it must agree with materializing the map and calling .max().
    for grid in (8, 33, 224):
        for size in ([1, 1], [5, 2], [grid - 2, 1]):
            for mode in ("corner", "edge"):
                params, size_arr = EnvParams(grid=grid), jnp.array(size)
                closed_form = float(regularity_max(params, size_arr, 2.0, mode))
                materialized = float(regularity_mask(params, size_arr, 2.0, mode).max())
                assert abs(closed_form - materialized) < 1e-6, (grid, size, mode)


def test_regularity_max_is_zero_when_macro_leaves_no_interior() -> None:
    # A macro that fills the canvas has an all-zero cost map; callers divide by this, so it
    # must report 0 rather than a bogus positive scale.
    params, size = EnvParams(grid=4), jnp.array([4, 4])
    assert float(regularity_max(params, size)) == 0.0
    assert float(regularity_mask(params, size).max()) == 0.0


def test_regularity_cost_reads_the_same_value_as_the_mask() -> None:
    # regularity_cost() skips building the map; it must still agree with it cell by cell.
    params, size = EnvParams(grid=24), jnp.array([3, 5])
    mask = regularity_mask(params, size, cell_size=7.0, mode="corner")
    for x, y in ((0, 0), (1, 1), (12, 8), (23, 23)):
        cost = regularity_cost(params, size, jnp.array([x, y]), cell_size=7.0, mode="corner")
        assert abs(float(cost) - float(mask[x, y])) < 1e-6, (x, y)


def test_regularity_mask_handles_non_square_grids_per_axis() -> None:
    # placax supports grid_y != grid_x (EXPlace does not), so each axis must use its own extent.
    params = EnvParams(grid=32, grid_y=8)
    mask = regularity_mask(params, jnp.array([1, 1]))
    assert mask.shape == (32, 8)
    # Along y=0 the y term is 0, so that column is the bare x cost, and vice versa. The y axis
    # is 4x shorter, so its own peak must be far smaller - a single shared grid extent would
    # make these equal, which is the bug this guards.
    x_axis_peak = float(mask[:, 0].max())
    y_axis_peak = float(mask[0, :].max())
    assert x_axis_peak == 15.0 and y_axis_peak == 3.0


def test_lookahead_regularity_masks_stacks_one_map_per_size() -> None:
    params = EnvParams(grid=16)
    sizes = jnp.array([[1, 1], [3, 2], [5, 5]])
    stacked = lookahead_regularity_masks(params, sizes)
    assert stacked.shape == (3, 16, 16)
    for i in range(3):
        assert jnp.allclose(stacked[i], regularity_mask(params, sizes[i]))
