from placax.core import reset  # noqa: F401  must precede jax imports
from placax.netlist.padding import build_macro_net_index  # noqa: F401
from placax.types import EnvParams, EnvState  # noqa: F401
from placax_agents.policy.observation import (  # noqa: F401
    lookahead_sizes,
    make_wiremask_observation,
    observation,
)

import jax.numpy as jnp


def test_observation_shapes_and_content() -> None:
    params = EnvParams(grid=8, n_macros=3)
    sizes_array = jnp.array([[2.0, 2.0], [1.0, 1.0], [3.0, 1.0]])
    state = reset(params)

    obs = observation(state, params, sizes_array)
    assert obs["canvas"].shape == (8, 8)
    assert not obs["canvas"].any()  # nothing placed yet
    assert obs["current_macro_size"].tolist() == [2.0, 2.0]  # macro 0's size, step=0


def test_observation_canvas_shows_placed_macros_only() -> None:
    params = EnvParams(grid=8, n_macros=3)
    sizes_array = jnp.array([[2.0, 2.0], [1.0, 1.0], [3.0, 1.0]])
    positions = jnp.array([[1, 1], [-1, -1], [-1, -1]])
    state = EnvState(positions=positions, step=1)
    obs = observation(state, params, sizes_array)
    assert obs["canvas"][1:3, 1:3].all()  # macro 0's 2x2 footprint
    assert obs["current_macro_size"].tolist() == [1.0, 1.0]  # macro 1's size, step=1


def test_observation_exposes_raw_facts_for_non_image_policies() -> None:
    # A GNN or any other non-image policy needs these, not canvas -
    # confirms the raw facts are genuinely present, not just canvas.
    params = EnvParams(grid=8, n_macros=3)
    sizes_array = jnp.array([[2.0, 2.0], [1.0, 1.0], [3.0, 1.0]])
    positions = jnp.array([[1, 1], [-1, -1], [-1, -1]])
    state = EnvState(positions=positions, step=1)
    obs = observation(state, params, sizes_array)

    assert obs["positions"].tolist() == positions.tolist()
    assert obs["sizes_array"].tolist() == sizes_array.tolist()
    assert obs["placed_mask"].tolist() == [True, False, False]
    assert int(obs["step"]) == 1


def test_observation_lookahead_defaults_to_one_step() -> None:
    params = EnvParams(grid=8, n_macros=3)
    sizes_array = jnp.array([[2.0, 2.0], [1.0, 1.0], [3.0, 1.0]])
    state = reset(params)
    obs = observation(state, params, sizes_array)
    assert obs["lookahead_sizes"].tolist() == [[2.0, 2.0]]  # same content as current_macro_size


def test_observation_lookahead_sees_multiple_upcoming_macros() -> None:
    params = EnvParams(grid=8, n_macros=3)
    sizes_array = jnp.array([[2.0, 2.0], [1.0, 1.0], [3.0, 1.0]])
    state = reset(params)
    obs = observation(state, params, sizes_array, lookahead=3)
    assert obs["lookahead_sizes"].tolist() == [[2.0, 2.0], [1.0, 1.0], [3.0, 1.0]]


def test_lookahead_sizes_zero_pads_past_the_last_macro() -> None:
    params = EnvParams(grid=8, n_macros=2)
    sizes_array = jnp.array([[2.0, 2.0], [1.0, 1.0]])
    state = EnvState(positions=jnp.full((2, 2), -1), step=1)  # only 1 macro left
    sizes = lookahead_sizes(state, params, sizes_array, horizon=3)
    assert sizes.tolist() == [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]


def test_make_wiremask_observation_adds_a_wiremask_key_without_dropping_the_base() -> None:
    params = EnvParams(grid=4, n_macros=2)
    sizes_array = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    positions = jnp.array([[1, 1], [-1, -1]])
    state = EnvState(positions=positions, step=1)

    padded_pin_idx = jnp.array([[0, 1]])
    padded_pin_offset = jnp.zeros((1, 2, 2))
    valid_mask = jnp.array([[True, True]])
    macro_net_idx, macro_net_offset, macro_net_valid = build_macro_net_index(
        padded_pin_idx, padded_pin_offset, valid_mask, n_macros=2
    )

    state_fn = make_wiremask_observation(
        padded_pin_idx, padded_pin_offset, valid_mask, macro_net_idx, macro_net_offset, macro_net_valid
    )
    obs = state_fn(state, params, sizes_array)
    assert obs["wiremask"].shape == (4, 4)
    assert obs["lookahead_wiremasks"].shape == (1, 4, 4)  # default lookahead=1
    assert obs["lookahead_wiremasks"][0].tolist() == obs["wiremask"].tolist()
    assert obs["canvas"].shape == (4, 4)  # base observation() keys still present


def test_make_wiremask_observation_lookahead_previews_multiple_macros() -> None:
    params = EnvParams(grid=4, n_macros=3)
    sizes_array = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    positions = jnp.array([[1, 1], [-1, -1], [-1, -1]])
    state = EnvState(positions=positions, step=1)

    padded_pin_idx = jnp.array([[0, 1], [1, 2]])
    padded_pin_offset = jnp.zeros((2, 2, 2))
    valid_mask = jnp.array([[True, True], [True, True]])
    macro_net_idx, macro_net_offset, macro_net_valid = build_macro_net_index(
        padded_pin_idx, padded_pin_offset, valid_mask, n_macros=3
    )

    state_fn = make_wiremask_observation(
        padded_pin_idx, padded_pin_offset, valid_mask, macro_net_idx, macro_net_offset, macro_net_valid,
        lookahead=2,
    )
    obs = state_fn(state, params, sizes_array)
    assert obs["lookahead_wiremasks"].shape == (2, 4, 4)
    assert obs["wiremask"].tolist() == obs["lookahead_wiremasks"][0].tolist()
