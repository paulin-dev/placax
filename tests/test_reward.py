from placax.types import EnvParams  # noqa: F401  must precede jax imports
from placax_agents.training.reward import make_expert_reward, make_scaled_hpwl_reward

import jax.numpy as jnp


def test_scaled_reward_matches_hand_calculation() -> None:
    # 2 macros connected by one net, grid positions (10,20) and (15,20),
    # cell_size=10, sizes (500,2136) and (100,100), real offsets applied.
    # Same numbers as the earlier verified to_real_centers example for
    # macro 0; macro 1 kept simple (offset 0) to make this hand-checkable.
    padded_pin_idx = jnp.array([[0, 1]])
    padded_pin_offset = jnp.array([[[-248.5, 16.0], [0.0, 0.0]]])
    valid_mask = jnp.array([[True, True]])
    sizes_array = jnp.array([[500.0, 2136.0], [100.0, 100.0]])
    cell_size = 10.0

    reward_fn = make_scaled_hpwl_reward(
        padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size
    )
    positions = jnp.array([[10, 20], [15, 20]])
    all_placed = jnp.array([True, True])
    reward = reward_fn(positions, positions, all_placed, all_placed)

    # macro 0 real center: (10*10+500/2, 20*10+2136/2) = (350, 1268)
    # macro 0 pin: (350-248.5, 1268+16) = (101.5, 1284.0)
    # macro 1 real center: (15*10+100/2, 20*10+100/2) = (200, 250)
    # macro 1 pin (no offset): (200, 250)
    # make_scaled_hpwl_reward quantizes each pin to the nearest cell_size=10 multiple before
    # computing HPWL (matching the reference's own per-step rounding - see hpwl()'s cell_size
    # doc): pin 0 -> (round(101.5/10)*10, round(1284.0/10)*10) = (100.0, 1280.0); pin 1 is
    # already on-lattice -> (200.0, 250.0) unchanged.
    # HPWL = |100.0-200.0| + |1280.0-250.0| = 100.0 + 1030.0 = 1130.0
    assert abs(float(reward) - (-1130.0)) < 1e-2


def test_expert_reward_with_zero_weight_reproduces_the_hpwl_reward_exactly() -> None:
    # The regularity term must be strictly opt-in: existing runs cannot shift under it.
    padded_pin_idx = jnp.array([[0, 1]])
    padded_pin_offset = jnp.array([[[-248.5, 16.0], [0.0, 0.0]]])
    valid_mask = jnp.array([[True, True]])
    sizes_array = jnp.array([[500.0, 2136.0], [100.0, 100.0]])
    params = EnvParams(grid=64, n_macros=2)

    args = (padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, 10.0)
    baseline = make_scaled_hpwl_reward(*args, dense=True)
    expert = make_expert_reward(*args, params, dense=True, regularity_weight=0.0)

    old_pos = jnp.array([[10, 20], [-1, -1]])
    new_pos = jnp.array([[10, 20], [15, 20]])
    old_placed, new_placed = jnp.array([True, False]), jnp.array([True, True])
    assert float(expert(old_pos, new_pos, old_placed, new_placed)) == float(
        baseline(old_pos, new_pos, old_placed, new_placed)
    )


def test_expert_reward_penalizes_the_canvas_center_more_than_a_corner() -> None:
    # Same netlist, same step, two candidate cells for macro 1: the corner should score higher.
    padded_pin_idx = jnp.array([[0, 1]])
    padded_pin_offset = jnp.zeros((1, 2, 2))
    valid_mask = jnp.array([[True, True]])
    sizes_array = jnp.array([[100.0, 100.0], [100.0, 100.0]])
    params = EnvParams(grid=64, n_macros=2)

    # Both terms must be on a comparable scale for the trade-off to mean anything, so use
    # run_maskplace's own reward_scale (1 / (cell_size * 200)) rather than raw real units -
    # at reward_scale=1.0 the HPWL term is ~3 orders of magnitude larger and no sane
    # regularity_weight can move the outcome. See make_expert_reward's docstring.
    scale = 1.0 / (10.0 * 200)
    args = (padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, 10.0)
    reward_fn = make_expert_reward(*args, params, dense=True, reward_scale=scale, regularity_weight=1.0)
    hpwl_only = make_scaled_hpwl_reward(*args, dense=True, reward_scale=scale)

    old_pos = jnp.array([[32, 32], [-1, -1]])
    old_placed, new_placed = jnp.array([True, False]), jnp.array([True, True])
    at_corner = old_pos.at[1].set(jnp.array([0, 0]))
    at_center = old_pos.at[1].set(jnp.array([31, 31]))

    # The corner is farther from macro 0, so it is *worse* on HPWL alone - the regularity term
    # has to be what flips the ordering, which is exactly the behaviour being asserted.
    assert float(hpwl_only(old_pos, at_corner, old_placed, new_placed)) < float(
        hpwl_only(old_pos, at_center, old_placed, new_placed)
    )
    assert float(reward_fn(old_pos, at_corner, old_placed, new_placed)) > float(
        reward_fn(old_pos, at_center, old_placed, new_placed)
    )


def test_expert_reward_regularity_term_is_normalized_to_unit_weight() -> None:
    # At the map's peak the normalized cost is 1.0, so the reward drops by exactly the weight.
    padded_pin_idx = jnp.array([[0, 0]])
    padded_pin_offset = jnp.zeros((1, 2, 2))
    valid_mask = jnp.array([[True, False]])  # no real net, so HPWL contributes nothing
    sizes_array = jnp.array([[100.0, 100.0], [100.0, 100.0]])
    params = EnvParams(grid=64, n_macros=2)

    args = (padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, 100.0)
    weight = 0.45
    reward_fn = make_expert_reward(*args, params, dense=True, regularity_weight=weight)
    baseline = make_scaled_hpwl_reward(*args, dense=True)

    old_pos = jnp.array([[0, 0], [-1, -1]])
    old_placed, new_placed = jnp.array([True, False]), jnp.array([True, True])
    # cell_size 100 vs. size 100 -> a 1x1 grid macro, whose cost map peaks at the exact centre.
    peak = jnp.array([32, 32])
    new_pos = old_pos.at[1].set(peak)
    delta = float(baseline(old_pos, new_pos, old_placed, new_placed)) - float(
        reward_fn(old_pos, new_pos, old_placed, new_placed)
    )
    assert abs(delta - weight) < 1e-5


def test_expert_reward_is_jittable_and_scannable() -> None:
    # The reward runs inside a jitted scan over the episode, so it must trace cleanly.
    import jax

    padded_pin_idx = jnp.array([[0, 1]])
    padded_pin_offset = jnp.zeros((1, 2, 2))
    valid_mask = jnp.array([[True, True]])
    sizes_array = jnp.array([[100.0, 100.0], [100.0, 100.0]])
    params = EnvParams(grid=32, n_macros=2)
    reward_fn = make_expert_reward(
        padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, 10.0,
        params, dense=True, regularity_weight=0.45,
    )
    old_pos = jnp.array([[4, 4], [-1, -1]])
    new_pos = jnp.array([[4, 4], [9, 9]])
    out = jax.jit(reward_fn)(old_pos, new_pos, jnp.array([True, False]), jnp.array([True, True]))
    assert jnp.isfinite(out)
