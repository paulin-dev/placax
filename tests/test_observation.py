from placax.core import reset  # noqa: F401  must precede jax imports
from placax.types import EnvParams  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401

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
    from placax.types import EnvState

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
    from placax.types import EnvState

    state = EnvState(positions=positions, step=1)
    obs = observation(state, params, sizes_array)

    assert obs["positions"].tolist() == positions.tolist()
    assert obs["sizes_array"].tolist() == sizes_array.tolist()
    assert obs["placed_mask"].tolist() == [True, False, False]
    assert int(obs["step"]) == 1
