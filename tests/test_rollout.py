import pathlib

from placax.core import reset  # noqa: F401  must precede jax imports
from placax.netlist import load_netlist  # noqa: F401
from placax.netlist.padding import build_padded_arrays  # noqa: F401
from placax.extras.rewards import make_hpwl_reward  # noqa: F401
from placax.types import EnvParams  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.policy.architectures.cnn import CNNActorCritic  # noqa: F401
from placax_agents.training.rollout import collect_rollout  # noqa: F401
from placax_agents.policy.scale import compute_grid_scale  # noqa: F401

import jax.numpy as jnp
import pytest
from jax import random

REAL_ADAPTEC1 = pathlib.Path("/home/claude/maskplace/maskplace/adaptec1")


def _make_toy_setup(params: EnvParams):
    sizes_array = jnp.array([[2.0, 2.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]][: params.n_macros])
    padded_pin_idx = jnp.array([[0, 1]])
    padded_pin_offset = jnp.zeros((1, 2, 2))
    valid_mask = jnp.array([[True, True]])
    reward_fn = make_hpwl_reward(padded_pin_idx, padded_pin_offset, valid_mask)
    return sizes_array, reward_fn


def test_collect_rollout_state_fn_is_genuinely_swappable() -> None:
    # Confirms state_fn is an actual, honored parameter, not just a
    # default-valued argument nobody can meaningfully override - a
    # custom state_fn's output should show up in the real trajectory.
    params = EnvParams(grid=8, n_macros=4)
    sizes_array, reward_fn = _make_toy_setup(params)
    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(init_key, obs0)

    def fake_state_fn(state, params, sizes_array):
        return {
            "canvas": jnp.ones((params.grid, params.grid), dtype=bool),
            "current_macro_size": sizes_array[state.step],
        }

    key, rollout_key = random.split(key)
    trajectory, _final_state = collect_rollout(
        rollout_key, variables, policy.apply, params, reward_fn, sizes_array,
        cell_size=1.0, state_fn=fake_state_fn,
    )
    assert bool(trajectory["obs"]["canvas"][0].all())  # matches fake_state_fn, not the real observation()


def test_collect_rollout_shapes_and_sparse_reward() -> None:
    params = EnvParams(grid=8, n_macros=4)
    sizes_array, reward_fn = _make_toy_setup(params)

    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(init_key, obs0)

    key, rollout_key = random.split(key)
    trajectory, final_state = collect_rollout(
        rollout_key, variables, policy.apply, params, reward_fn, sizes_array, cell_size=1.0
    )

    for name, expected_shape in [
        ("action", (4, 2)),
        ("done", (4,)),
        ("log_prob", (4,)),
        ("reward", (4,)),
        ("value", (4,)),
    ]:
        assert trajectory[name].shape == expected_shape
    assert trajectory["obs"]["canvas"].shape == (4, 8, 8)

    assert trajectory["done"].tolist() == [False, False, False, True]
    assert trajectory["reward"][:3].tolist() == [0.0, 0.0, 0.0]  # sparse: 0 until done
    assert final_state.step == 4
    assert not (final_state.positions == -1).any()


def test_collect_rollout_final_reward_matches_hand_calculation() -> None:
    # Net connects macro 0 and macro 1 only - final reward should be
    # exactly -HPWL of their final positions, computable by hand.
    params = EnvParams(grid=8, n_macros=4)
    sizes_array, reward_fn = _make_toy_setup(params)

    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(init_key, obs0)

    key, rollout_key = random.split(key)
    trajectory, final_state = collect_rollout(
        rollout_key, variables, policy.apply, params, reward_fn, sizes_array, cell_size=1.0
    )

    pos0, pos1 = final_state.positions[0], final_state.positions[1]
    expected_hpwl = float(jnp.abs(pos0[0] - pos1[0]) + jnp.abs(pos0[1] - pos1[1]))
    assert float(trajectory["reward"][-1]) == -expected_hpwl


@pytest.mark.skipif(not REAL_ADAPTEC1.exists(), reason="real adaptec1 benchmark not available")
def test_collect_rollout_at_real_scale() -> None:
    macro_sizes, nets = load_netlist(REAL_ADAPTEC1)
    _, sizes_array, padded_pin_idx, padded_pin_offset, valid_mask = build_padded_arrays(
        macro_sizes, nets
    )
    reward_fn = make_hpwl_reward(padded_pin_idx, padded_pin_offset, valid_mask)
    params = EnvParams(grid=64, n_macros=len(macro_sizes))
    cell_size = compute_grid_scale(sizes_array, params.grid)

    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(init_key, obs0)

    key, rollout_key = random.split(key)
    trajectory, final_state = collect_rollout(
        rollout_key, variables, policy.apply, params, reward_fn, sizes_array, cell_size
    )

    assert final_state.step == 543
    assert not (final_state.positions == -1).any()
    assert trajectory["done"].tolist() == [False] * 542 + [True]
    assert trajectory["reward"][-1] < 0  # a real, nonzero (negative) HPWL-based reward
