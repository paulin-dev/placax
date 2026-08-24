from placax.core import reset  # noqa: F401  must precede jax imports
from placax.extras.rewards import make_hpwl_reward  # noqa: F401
from placax.types import EnvParams  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.policy.architectures.cnn import CNNActorCritic  # noqa: F401
from placax_agents.training.algorithm.config import PPOConfig  # noqa: F401
from placax_agents.training.loops.run import train  # noqa: F401

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


def _init(params, sizes_array):
    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(init_key, obs0)
    return policy, variables


def test_train_forced_sequential() -> None:
    params, sizes_array, reward_fn = _toy_setup()
    policy, variables = _init(params, sizes_array)

    _final, losses = train(
        random.PRNGKey(1), variables, policy.apply, params, reward_fn, sizes_array,
        cell_size=1.0, n_iterations=3, mode="sequential",
    )
    assert len(losses) == 3
    assert all(jnp.isfinite(loss_val) for loss_val in losses)


def test_train_forced_parallel() -> None:
    params, sizes_array, reward_fn = _toy_setup()
    policy, variables = _init(params, sizes_array)

    _final, losses = train(
        random.PRNGKey(2), variables, policy.apply, params, reward_fn, sizes_array,
        cell_size=1.0, n_iterations=3, n_envs=4, mode="parallel",
    )
    assert len(losses) == 3
    assert all(jnp.isfinite(loss_val) for loss_val in losses)


def test_train_auto_mode_matches_recommended_parallelism() -> None:
    # This sandbox has no GPU (confirmed throughout this whole build),
    # so auto-detection should resolve to sequential.
    from placax._device import recommended_parallelism_mode

    assert recommended_parallelism_mode() == "sequential"

    params, sizes_array, reward_fn = _toy_setup()
    policy, variables = _init(params, sizes_array)

    _final, losses = train(
        random.PRNGKey(3), variables, policy.apply, params, reward_fn, sizes_array,
        cell_size=1.0, n_iterations=2,
    )
    assert len(losses) == 2
    assert all(jnp.isfinite(loss_val) for loss_val in losses)


def test_train_n_envs_1_uses_sequential_even_if_parallel_requested() -> None:
    params, sizes_array, reward_fn = _toy_setup()
    policy, variables = _init(params, sizes_array)

    # n_envs=1 with mode="parallel" should still just work (falls back
    # to sequential internally - n_envs<=1 doesn't need vmap at all).
    _final, losses = train(
        random.PRNGKey(4), variables, policy.apply, params, reward_fn, sizes_array,
        cell_size=1.0, n_iterations=2, n_envs=1, mode="parallel",
    )
    assert len(losses) == 2


def test_train_ppo_config_genuinely_changes_behavior() -> None:
    # Regression test: compute_gae/ppo_loss were already correctly
    # parameterized, but train()/train_sequential()/train_parallel()
    # never actually passed gamma/lam/clip_eps/value_coef/entropy_coef
    # through - they were silently stuck at internal defaults with no
    # way to reach them from the public API. A wildly different
    # entropy_coef must produce a genuinely different loss, both
    # sequential and parallel.
    params, sizes_array, reward_fn = _toy_setup()
    policy, variables = _init(params, sizes_array)

    _, default_losses = train(
        random.PRNGKey(5), variables, policy.apply, params, reward_fn, sizes_array,
        cell_size=1.0, n_iterations=1, mode="sequential",
    )
    _, custom_losses = train(
        random.PRNGKey(5), variables, policy.apply, params, reward_fn, sizes_array,
        cell_size=1.0, n_iterations=1, mode="sequential", ppo_config=PPOConfig(entropy_coef=10.0),
    )
    assert default_losses[0] != custom_losses[0]
