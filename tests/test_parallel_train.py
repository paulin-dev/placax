from placax.core import reset  # noqa: F401  must precede jax imports
from placax.extras.rewards import make_hpwl_reward  # noqa: F401
from placax.types import EnvParams  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.training.loops.parallel_train import train_parallel as parallel_train  # noqa: F401
from placax_agents.policy.architectures.cnn import CNNActorCritic  # noqa: F401

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


def test_parallel_train_produces_finite_losses_and_changes_params() -> None:
    params, sizes_array, reward_fn = _toy_setup()
    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(init_key, obs0)
    initial_leaves = jax.tree_util.tree_leaves(variables)

    key, train_key = random.split(key)
    final_variables, losses = parallel_train(
        train_key, variables, policy.apply, params, reward_fn, sizes_array,
        cell_size=1.0, n_envs=4, n_iterations=3,
    )

    assert len(losses) == 3
    assert all(jnp.isfinite(loss_val) for loss_val in losses)

    final_leaves = jax.tree_util.tree_leaves(final_variables)
    total_change = sum(jnp.sum(jnp.abs(a - b)) for a, b in zip(initial_leaves, final_leaves))
    assert total_change > 0
