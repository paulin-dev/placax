from placax.core import reset  # noqa: F401  must precede jax imports
from placax.extras.rewards import make_hpwl_reward  # noqa: F401
from placax.types import EnvParams  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.policy.architectures.cnn import CNNActorCritic  # noqa: F401
from placax_agents.training.loops.train import train_sequential as train  # noqa: F401

import jax
import jax.numpy as jnp
from jax import random


def _toy_setup():
    params = EnvParams(grid=8, n_macros=4)
    sizes_array = jnp.array([[2.0, 2.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    padded_pin_idx = jnp.array([[0, 1]])
    padded_pin_offset = jnp.zeros((1, 2, 2))
    valid_mask = jnp.array([[True, True]])
    reward_fn = make_hpwl_reward(padded_pin_idx, padded_pin_offset, valid_mask)
    return params, sizes_array, reward_fn


def test_train_produces_finite_losses_and_changes_params() -> None:
    params, sizes_array, reward_fn = _toy_setup()
    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(init_key, obs0)
    initial_leaves = jax.tree_util.tree_leaves(variables)

    key, train_key = random.split(key)
    final_variables, losses = train(
        train_key, variables, policy.apply, params, reward_fn, sizes_array,
        cell_size=1.0, n_iterations=5,
    )

    assert len(losses) == 5
    assert all(jnp.isfinite(loss_val) for loss_val in losses)

    final_leaves = jax.tree_util.tree_leaves(final_variables)
    total_change = sum(jnp.sum(jnp.abs(a - b)) for a, b in zip(initial_leaves, final_leaves))
    assert total_change > 0


def test_train_sequential_checkpoint_path_interrupted_matches_continuous(tmp_path) -> None:
    # The same critical property resumable_train guarantees, now for
    # the simpler train_sequential(checkpoint_path=...) path: splitting
    # a run across two calls sharing the same checkpoint_path must give
    # bit-for-bit identical results to one continuous call.
    params, sizes_array, reward_fn = _toy_setup()
    policy = CNNActorCritic()
    obs0 = observation(reset(params), params, sizes_array)
    variables_init = policy.init(random.PRNGKey(0), obs0)

    path_interrupted = tmp_path / "interrupted.bin"
    _v1, losses1 = train(
        random.PRNGKey(1), variables_init, policy.apply, params, reward_fn, sizes_array,
        cell_size=1.0, n_iterations=3, checkpoint_path=path_interrupted,
    )
    v_interrupted, losses2 = train(
        random.PRNGKey(1), variables_init, policy.apply, params, reward_fn, sizes_array,
        cell_size=1.0, n_iterations=2, checkpoint_path=path_interrupted,
    )
    interrupted_losses = losses1 + losses2

    path_continuous = tmp_path / "continuous.bin"
    v_continuous, continuous_losses = train(
        random.PRNGKey(1), variables_init, policy.apply, params, reward_fn, sizes_array,
        cell_size=1.0, n_iterations=5, checkpoint_path=path_continuous,
    )

    assert interrupted_losses == continuous_losses
    leaves_interrupted = jax.tree_util.tree_leaves(v_interrupted)
    leaves_continuous = jax.tree_util.tree_leaves(v_continuous)
    assert all((a == b).all() for a, b in zip(leaves_interrupted, leaves_continuous))


def test_train_at_real_scale_produces_finite_losses() -> None:
    import pathlib

    import pytest

    from placax.netlist import load_netlist
    from placax.netlist.padding import build_padded_arrays
    from placax_agents.policy.scale import compute_grid_scale

    adaptec1_dir = pathlib.Path("/home/claude/maskplace/maskplace/adaptec1")
    if not adaptec1_dir.exists():
        pytest.skip("real adaptec1 benchmark not available")

    macro_sizes, nets = load_netlist(adaptec1_dir)
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

    key, train_key = random.split(key)
    _final_variables, losses = train(
        train_key, variables, policy.apply, params, reward_fn, sizes_array,
        cell_size, n_iterations=1,
    )
    assert jnp.isfinite(losses[0])
    # Regression guard: before advantage/return normalization, real-scale
    # losses were in the hundreds of millions (real HPWL-scale returns,
    # squared, dominate the value loss). A sane loss should be small.
    assert abs(losses[0]) < 100
