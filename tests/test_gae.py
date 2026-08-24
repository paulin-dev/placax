from placax_agents.training.algorithm.gae import compute_gae  # noqa: F401  must precede jax imports

import jax.numpy as jnp


def test_compute_gae_undiscounted_sparse_reward() -> None:
    # gamma=lam=1 (no discounting): with a single terminal reward and no
    # other rewards, the return-to-go from ANY point in the trajectory
    # is just that terminal reward - a clean, hand-verifiable property.
    rewards = jnp.array([0.0, 0.0, -10.0])
    values = jnp.array([1.0, 2.0, 3.0])
    dones = jnp.array([False, False, True])
    next_value = jnp.array(0.0)

    advantages, returns = compute_gae(rewards, values, dones, next_value, gamma=1.0, lam=1.0)
    assert advantages.tolist() == [-11.0, -12.0, -13.0]
    assert returns.tolist() == [-10.0, -10.0, -10.0]


def test_compute_gae_single_step() -> None:
    rewards = jnp.array([-5.0])
    values = jnp.array([2.0])
    dones = jnp.array([True])
    next_value = jnp.array(0.0)

    advantages, returns = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
    # delta = reward + gamma*next_value*(1-done) - value = -5 + 0 - 2 = -7
    # done=True kills both the bootstrap and the recursive term
    assert advantages.tolist() == [-7.0]
    assert returns.tolist() == [-5.0]  # advantage + value = -7 + 2 = -5, matches the raw reward
