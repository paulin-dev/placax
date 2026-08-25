from placax_agents.training.reward import make_scaled_hpwl_reward  # noqa: F401  must precede jax imports

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
    # HPWL = |101.5-200| + |1284.0-250| = 98.5 + 1034.0 = 1132.5
    assert abs(float(reward) - (-1132.5)) < 1e-2
