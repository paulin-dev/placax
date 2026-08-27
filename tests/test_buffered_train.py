from placax.core import reset  # noqa: F401  must precede jax imports
from placax.extras.rewards import make_hpwl_reward  # noqa: F401
from placax.types import EnvParams  # noqa: F401
from placax_agents.policy.architectures.cnn import CNNActorCritic  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.training.algorithm.config import PPOConfig  # noqa: F401
from placax_agents.training.algorithm.gae import compute_gae  # noqa: F401
from placax_agents.training.loops.buffered_train import (  # noqa: F401
    collect_buffer,
    train_buffered,
)

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


def test_collect_buffer_flattens_n_episodes_into_one_leading_dimension() -> None:
    params, sizes_array, reward_fn = _toy_setup()
    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(key, obs0)

    buffer = collect_buffer(key, variables, policy.apply, params, reward_fn, sizes_array, 1.0, n_episodes=3)
    assert buffer["reward"].shape == (3 * params.n_macros,)
    assert buffer["done"].shape == (3 * params.n_macros,)
    # done is True exactly once per episode - at each episode's last step.
    assert int(buffer["done"].sum()) == 3


def test_gae_on_a_concatenated_buffer_matches_gae_computed_per_episode() -> None:
    # The correctness property buffered training relies on: since done=True
    # firewalls every episode boundary, running compute_gae() ONCE over a
    # multi-episode concatenation must give identical per-episode results
    # to running it separately on each episode - not an approximation.
    n_macros = 4
    key = random.PRNGKey(1)
    reward = random.uniform(key, (2, n_macros))
    value = random.uniform(random.fold_in(key, 1), (2, n_macros))
    done = jnp.array([[False, False, False, True], [False, False, False, True]])

    separate = [
        compute_gae(reward[i], value[i], done[i], next_value=jnp.array(0.0))
        for i in range(2)
    ]
    separate_adv = jnp.concatenate([a for a, _r in separate])
    separate_ret = jnp.concatenate([r for _a, r in separate])

    concatenated_adv, concatenated_ret = compute_gae(
        reward.reshape(-1), value.reshape(-1), done.reshape(-1), next_value=jnp.array(0.0)
    )

    assert jnp.allclose(concatenated_adv, separate_adv, atol=1e-5)
    assert jnp.allclose(concatenated_ret, separate_ret, atol=1e-5)


def test_train_buffered_produces_finite_losses_and_changes_params() -> None:
    params, sizes_array, reward_fn = _toy_setup()
    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(init_key, obs0)
    # Copy, not just reference: train_buffered's minibatch update donates its variables
    # input for memory efficiency, so the original buffers get deleted once consumed.
    initial_leaves = [leaf.copy() for leaf in jax.tree_util.tree_leaves(variables)]

    key, train_key = random.split(key)
    final_variables, losses = train_buffered(
        train_key, variables, policy.apply, params, reward_fn, sizes_array, cell_size=1.0,
        n_iterations=2, n_episodes=3, ppo_epochs=2, batch_size=4,
    )

    assert len(losses) == 2
    assert all(jnp.isfinite(loss_val) for loss_val in losses)

    final_leaves = jax.tree_util.tree_leaves(final_variables)
    total_change = sum(jnp.sum(jnp.abs(a - b)) for a, b in zip(initial_leaves, final_leaves))
    assert total_change > 0


def test_train_buffered_accepts_an_extra_illegal_fn() -> None:
    params, sizes_array, reward_fn = _toy_setup()
    policy = CNNActorCritic()
    key = random.PRNGKey(3)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(key, obs0)

    def no_extra_restriction(obs):
        return jnp.zeros((params.grid, params.grid), dtype=bool)

    _final_variables, losses = train_buffered(
        key, variables, policy.apply, params, reward_fn, sizes_array, cell_size=1.0,
        n_iterations=1, n_episodes=2, ppo_epochs=1, batch_size=2,
        extra_illegal_fn=no_extra_restriction,
    )
    assert len(losses) == 1
    assert jnp.isfinite(losses[0])


def test_train_buffered_matches_maskplace_defaults_shape() -> None:
    # maskplace_ppo_config() + train_buffered's own defaults together
    # reproduce MaskPlace's procedure shape (n_episodes stands in for its
    # buffer_capacity = 10 * placed_num_macro, in effect, at n_episodes=10).
    from scripts.run_maskplace import maskplace_ppo_config

    params, sizes_array, reward_fn = _toy_setup()
    policy = CNNActorCritic()
    key = random.PRNGKey(2)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(key, obs0)

    _final_variables, losses = train_buffered(
        key, variables, policy.apply, params, reward_fn, sizes_array, cell_size=1.0,
        n_iterations=1, n_episodes=2, ppo_epochs=1, batch_size=2,
        ppo_config=maskplace_ppo_config(),
    )
    assert len(losses) == 1
    assert jnp.isfinite(losses[0])
