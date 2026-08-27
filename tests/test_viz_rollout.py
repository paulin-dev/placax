import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp

from placax.types import EnvParams
from placax_agents.policy.observation import observation
from placax_viz.rollout import collect_placement_history


def _fake_policy_apply(_variables, obs):
    grid_x, grid_y = obs["canvas"].shape
    return jnp.zeros((grid_x, grid_y)), jnp.array(0.0)


def test_collect_placement_history_places_every_macro_without_overlap() -> None:
    params = EnvParams(grid=8, n_macros=3)
    sizes_array = jnp.array([[2.0, 2.0], [1.0, 1.0], [2.0, 1.0]])

    history = collect_placement_history(
        None, _fake_policy_apply, params, sizes_array, cell_size=1.0, state_fn=observation
    )

    assert len(history) == params.n_macros + 1  # one entry per macro placed, plus the initial all-unplaced state
    assert (history[0] < 0).all()  # nothing placed yet
    assert (history[-1] >= 0).all()  # every macro placed by the end
    # positions strictly grow one macro at a time
    for step, snapshot in enumerate(history):
        assert (snapshot[:step] >= 0).all()
        assert (snapshot[step:] < 0).all()
