from placax.core import reset  # noqa: F401  must precede jax imports
from placax.extras.rewards import make_hpwl_reward  # noqa: F401
from placax.types import EnvParams  # noqa: F401
from placax_agents.training.algorithm.gae import compute_gae  # noqa: F401
from placax_agents.training.algorithm.loss import _entropy, ppo_loss  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.policy.architectures.cnn import CNNActorCritic  # noqa: F401
from placax_agents.training.rollout import collect_rollout  # noqa: F401

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


def test_entropy_gradient_is_not_nan_with_masked_logits() -> None:
    # Regression test: entropy's gradient through masked (-inf) logits
    # produced NaN before being fixed - JAX's where-gradient pitfall,
    # where guarding only the output of a multiplication still
    # differentiates through the discarded -inf branch.
    def make_masked_logits(raw_logits):
        illegal = jnp.array([[False, True], [False, False]])
        return jnp.where(illegal, -jnp.inf, raw_logits)

    raw = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    grad = jax.grad(lambda r: _entropy(make_masked_logits(r)))(raw)
    assert not jnp.isnan(grad).any()


def test_ppo_loss_is_finite() -> None:
    params, sizes_array, reward_fn = _toy_setup()
    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(init_key, obs0)

    key, rollout_key = random.split(key)
    trajectory, _final_state = collect_rollout(
        rollout_key, variables, policy.apply, params, reward_fn, sizes_array, cell_size=1.0
    )
    advantages, returns = compute_gae(
        trajectory["reward"], trajectory["value"], trajectory["done"], next_value=jnp.array(0.0)
    )

    loss = ppo_loss(variables, policy.apply, trajectory, advantages, returns, sizes_array, 1.0, params)
    assert jnp.isfinite(loss)


def test_ppo_loss_gradient_is_finite_and_nonzero() -> None:
    params, sizes_array, reward_fn = _toy_setup()
    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(init_key, obs0)

    key, rollout_key = random.split(key)
    trajectory, _final_state = collect_rollout(
        rollout_key, variables, policy.apply, params, reward_fn, sizes_array, cell_size=1.0
    )
    advantages, returns = compute_gae(
        trajectory["reward"], trajectory["value"], trajectory["done"], next_value=jnp.array(0.0)
    )

    grads = jax.grad(ppo_loss)(
        variables, policy.apply, trajectory, advantages, returns, sizes_array, 1.0, params
    )
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.isfinite(g).all() for g in leaves)
    grad_norm = sum(jnp.sum(g**2) for g in leaves) ** 0.5
    assert grad_norm > 0
