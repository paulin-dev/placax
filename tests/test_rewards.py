from placax.netlist.padding import build_macro_net_index  # noqa: F401  must precede jax imports
from placax.extras.rewards import hpwl, make_hpwl_reward, wiremask  # noqa: F401
from placax.types import EnvParams, EnvState  # noqa: F401

import jax.numpy as jnp

# 4 macros: 0=(0,0) 1=(3,0) 2=(3,2) 3=(0,2)
POSITIONS = jnp.array([[0, 0], [3, 0], [3, 2], [0, 2]])

# Net A: macros 0,1 (2 pins). Net B: macros 1,2,3 (3 pins). Padded to
# max_pins_per_net=3; Net A's padding slot points at macro 2 (an extreme
# corner) so an unmasked bug inflates the bbox instead of staying silent.
PADDED_PIN_IDX = jnp.array([[0, 1, 2], [1, 2, 3]])
VALID_MASK = jnp.array([[True, True, False], [True, True, True]])
ZERO_OFFSET = jnp.zeros((2, 3, 2))


def test_hpwl_matches_hand_calculation() -> None:
    # Net A: 3 wide, 0 tall -> 3. Net B: 3 wide, 2 tall -> 5. Total: 8.
    assert hpwl(POSITIONS, PADDED_PIN_IDX, ZERO_OFFSET, VALID_MASK) == 8.0


def test_unmasked_padding_would_silently_corrupt_result() -> None:
    all_valid = jnp.ones_like(VALID_MASK, dtype=bool)
    wrong = hpwl(POSITIONS, PADDED_PIN_IDX, ZERO_OFFSET, all_valid)
    assert wrong == 10.0  # Net A wrongly picks up macro 2 -> 5+5, not 3+5
    assert wrong != hpwl(POSITIONS, PADDED_PIN_IDX, ZERO_OFFSET, VALID_MASK)


def test_reward_is_negative_hpwl() -> None:
    reward_fn = make_hpwl_reward(PADDED_PIN_IDX, ZERO_OFFSET, VALID_MASK)
    assert reward_fn(POSITIONS) == -8.0


def test_nonzero_offset_changes_result_correctly() -> None:
    # Same macros/nets as the hand calculation above, but macro 1's pin on
    # Net A now sits 2 units further right than its macro's center - the
    # real, worked example from the conversation: an off-center pin
    # genuinely changes HPWL, not just theoretically.
    offset = ZERO_OFFSET.at[0, 1].set(jnp.array([2.0, 0.0]))  # Net A, pin slot 1 (macro 1)
    result = hpwl(POSITIONS, PADDED_PIN_IDX, offset, VALID_MASK)
    # Net A's real pin positions: macro0=(0,0), macro1=(3,0)+(2,0)=(5,0)
    # Net A width becomes 5 (not 3), height still 0 -> Net A contributes 5
    # Net B unchanged -> contributes 5. Total: 10, not the original 8.
    assert result == 10.0
    assert result != hpwl(POSITIONS, PADDED_PIN_IDX, ZERO_OFFSET, VALID_MASK)


def test_wiremask_matches_hand_calculation() -> None:
    # 2 macros, macro 0 already placed at (1,1), macro 1 about to be
    # placed (state.step=1), one net connecting them, no offsets.
    params = EnvParams(grid=4, n_macros=2)
    positions = jnp.array([[1, 1], [-1, -1]])
    state = EnvState(positions=positions, step=1)
    padded_pin_idx = jnp.array([[0, 1]])
    padded_pin_offset = jnp.zeros((1, 2, 2))
    valid_mask = jnp.array([[True, True]])
    macro_net_idx, macro_net_offset, macro_net_valid = build_macro_net_index(
        padded_pin_idx, padded_pin_offset, valid_mask, n_macros=2
    )

    wm = wiremask(
        state, params, padded_pin_idx, padded_pin_offset, valid_mask,
        macro_net_idx, macro_net_offset, macro_net_valid,
    )
    assert wm[1, 1] == 0.0  # same position as macro 0 - zero added wirelength
    assert wm[3, 3] == 4.0  # |3-1| + |3-1|
    assert wm[0, 0] == 2.0  # |0-1| + |0-1|


def test_wiremask_excludes_not_yet_placed_macros() -> None:
    # 3 macros: macro 0 placed at (0,0), macro 1 about to be placed
    # (step=1), macro 2 not yet placed. A net connecting macro 1 to macro
    # 2 (not yet placed) should contribute nothing to the wiremask -
    # macro 2's real position isn't known yet.
    params = EnvParams(grid=4, n_macros=3)
    positions = jnp.array([[0, 0], [-1, -1], [-1, -1]])
    state = EnvState(positions=positions, step=1)
    padded_pin_idx = jnp.array([[1, 2]])  # net connects macro 1 and macro 2 only
    padded_pin_offset = jnp.zeros((1, 2, 2))
    valid_mask = jnp.array([[True, True]])
    macro_net_idx, macro_net_offset, macro_net_valid = build_macro_net_index(
        padded_pin_idx, padded_pin_offset, valid_mask, n_macros=3
    )

    wm = wiremask(
        state, params, padded_pin_idx, padded_pin_offset, valid_mask,
        macro_net_idx, macro_net_offset, macro_net_valid,
    )
    # this net has fewer than 2 "relevant" pins (macro 2 excluded), so it
    # contributes nothing anywhere on the map - should be all zeros
    assert (wm == 0.0).all()


def test_wiremask_at_real_scale_uses_vmap_without_running_out_of_memory() -> None:
    # Regression test: the first version batched every one of grid*grid
    # candidate cells simultaneously via vmap over the WHOLE netlist,
    # meaning hpwl()'s internal gather (shaped (n_nets, max_pins)) got
    # materialized once per candidate cell, all at once - ~29.8GB on
    # real adaptec1 scale (64x64 grid), a genuine crash. The reverse
    # index (build_macro_net_index) fixes this by only touching each
    # macro's own small participation list per candidate, restoring
    # real vmap parallelism (not lax.map, which works but sequentially).
    import pathlib
    import time

    import numpy as np

    from placax.netlist import load_netlist
    from placax.netlist.padding import build_padded_arrays

    adaptec1_dir = pathlib.Path("/home/claude/maskplace/maskplace/adaptec1")
    if not adaptec1_dir.exists():
        import pytest

        pytest.skip("real adaptec1 benchmark not available")

    macro_sizes, nets = load_netlist(adaptec1_dir)
    _name_to_idx, _sizes, padded_pin_idx, padded_pin_offset, valid_mask = build_padded_arrays(
        macro_sizes, nets
    )
    macro_net_idx, macro_net_offset, macro_net_valid = build_macro_net_index(
        padded_pin_idx, padded_pin_offset, valid_mask, n_macros=len(macro_sizes)
    )
    params = EnvParams(grid=64, n_macros=len(macro_sizes))
    positions = np.full((params.n_macros, 2), -1)
    positions[0] = [10, 10]
    state = EnvState(positions=jnp.array(positions), step=1)

    t0 = time.perf_counter()
    wm = wiremask(
        state, params, padded_pin_idx, padded_pin_offset, valid_mask,
        macro_net_idx, macro_net_offset, macro_net_valid,
    )
    elapsed = time.perf_counter() - t0

    assert wm.shape == (64, 64)
    assert (wm >= 0.0).all()  # HPWL increase from a placement is never negative
    assert elapsed < 5.0  # generous bound; the vmap version should be fast, not just non-crashing
