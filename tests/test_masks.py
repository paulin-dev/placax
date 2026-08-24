from placax.extras.masks import boundary_mask, compute_occupied, occupancy_mask  # noqa: F401
from placax.types import EnvParams  # noqa: F401  both must precede jax imports

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


def test_boundary_mask_rectangular_grid() -> None:
    # Regression test: an earlier version assumed a square grid
    # everywhere - real chip die areas aren't always square.
    params = EnvParams(grid=4, grid_y=6)
    mask = boundary_mask(params, macro_size=(2, 2))
    assert mask.shape == (4, 6)
    assert bool(mask[3, :].all())  # last column, w=2 overflows grid_x=4
    assert bool(mask[:, 5].all())  # last row, h=2 overflows grid_y=6
    assert not bool(mask[0, 0])  # fits fine
