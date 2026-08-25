"""Buffer + minibatch-epoch PPO training: collect n_episodes of rollout
into one buffer, then run ppo_epochs passes of shuffled batch_size
minibatch updates over it - a third training-loop shape alongside
train_sequential (one episode, one full-batch update) and train_parallel
(n_envs episodes, one full-batch update). Matches MaskPlace's own PPO2.py
procedure (buffer_capacity/ppo_epoch/batch_size), generalized: n_episodes
takes the place of its `10 * placed_num_macro`-derived buffer size."""
import pathlib

from placax.types import EnvParams, RewardFn  # must precede jax imports
from placax_agents.policy.observation import observation
from placax_agents.training.algorithm.config import PPOConfig
from placax_agents.training.algorithm.gae import compute_gae
from placax_agents.training.algorithm.loss import ppo_loss
from placax_agents.training.algorithm.optimizer_step import apply_gradient_update
from placax_agents.training.algorithm.running_stats import RunningStats
from placax_agents.training.loops.common import checkpoint_every_n, make_step_input, open_train_state
from placax_agents.training.rollout import collect_rollout
from placax_agents.types import AlgorithmFn, ExtraIllegalFn, StateFn

import jax
import jax.numpy as jnp
import optax


def collect_buffer(
    key: jax.Array,
    variables,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    reward_fn: RewardFn,
    sizes_array: jax.Array,
    cell_size: float,
    n_episodes: int,
    state_fn: StateFn = observation,
    extra_illegal_fn: ExtraIllegalFn | None = None,
):
    """n_episodes of collect_rollout, vmapped and flattened into one
    buffer of (n_episodes * n_macros) transitions, in temporal order per
    episode - GAE's done-flag reset means compute_gae() below is safe to
    run once over the whole concatenation, not once per episode."""
    keys = jax.random.split(key, n_episodes)
    trajectories, _ = jax.vmap(
        collect_rollout, in_axes=(0, None, None, None, None, None, None, None, None)
    )(keys, variables, policy_apply_fn, params, reward_fn, sizes_array, cell_size, state_fn, extra_illegal_fn)
    flatten = lambda x: x.reshape((-1,) + x.shape[2:])  # noqa: E731  (n_episodes, n_macros, ...) -> (n, ...)
    return jax.tree_util.tree_map(flatten, trajectories)


def _minibatch_update(
    variables, opt_state, running_stats, optimizer, policy_apply_fn, params,
    batch_trajectory, batch_advantages, batch_returns, cell_size, ppo_config, extra_illegal_fn,
):
    """One gradient step over one minibatch slice of an already-built buffer."""
    def loss_fn(policy_params, normalized_advantages, normalized_returns):
        return ppo_loss(
            policy_params, policy_apply_fn, batch_trajectory, normalized_advantages, normalized_returns,
            cell_size, params,
            clip_eps=ppo_config.clip_eps, value_coef=ppo_config.value_coef,
            entropy_coef=ppo_config.entropy_coef, value_loss_fn=ppo_config.value_loss_fn,
            extra_illegal_fn=extra_illegal_fn,
        )

    return apply_gradient_update(
        variables, opt_state, running_stats, optimizer, loss_fn, batch_advantages, batch_returns
    )


_jitted_minibatch_update = jax.jit(
    _minibatch_update, static_argnames=("optimizer", "policy_apply_fn", "ppo_config", "extra_illegal_fn")
)


def _shuffled_batches(key: jax.Array, buffer_size: int, batch_size: int) -> jax.Array:
    """(n_batches, batch_size) int array of shuffled buffer indices - a
    remainder shorter than batch_size is dropped, matching MaskPlace's
    own BatchSampler(..., drop_last=True)."""
    n_batches = buffer_size // batch_size
    perm = jax.random.permutation(key, buffer_size)
    return perm[: n_batches * batch_size].reshape(n_batches, batch_size)


def buffered_train_step(
    key: jax.Array,
    variables,
    opt_state,
    running_stats: RunningStats,
    optimizer: optax.GradientTransformation,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    reward_fn: RewardFn,
    sizes_array: jax.Array,
    cell_size: float,
    n_episodes: int,
    ppo_epochs: int,
    batch_size: int,
    state_fn: StateFn = observation,
    ppo_config: PPOConfig = PPOConfig(),
    extra_illegal_fn: ExtraIllegalFn | None = None,
):
    """One full buffer-collect + multi-epoch-minibatch update cycle.
    Returns (variables, opt_state, running_stats, last_minibatch_loss)."""
    key, buffer_key = jax.random.split(key)
    buffer = collect_buffer(
        buffer_key, variables, policy_apply_fn, params, reward_fn, sizes_array, cell_size, n_episodes,
        state_fn, extra_illegal_fn,
    )
    advantages, returns = compute_gae(
        buffer["reward"], buffer["value"], buffer["done"], next_value=jnp.array(0.0),
        gamma=ppo_config.gamma, lam=ppo_config.lam,
    )

    loss = jnp.array(0.0)
    for _epoch in range(ppo_epochs):
        key, shuffle_key = jax.random.split(key)
        for batch_idx in _shuffled_batches(shuffle_key, advantages.shape[0], batch_size):
            batch_trajectory = jax.tree_util.tree_map(lambda x: x[batch_idx], buffer)
            variables, opt_state, running_stats, loss = _jitted_minibatch_update(
                variables, opt_state, running_stats, optimizer, policy_apply_fn, params,
                batch_trajectory, advantages[batch_idx], returns[batch_idx], cell_size, ppo_config,
                extra_illegal_fn,
            )
    return variables, opt_state, running_stats, loss


def train_buffered(
    key: jax.Array,
    variables,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    reward_fn: RewardFn,
    sizes_array: jax.Array,
    cell_size: float,
    n_iterations: int,
    n_episodes: int = 10,
    ppo_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    state_fn: StateFn = observation,
    optimizer: optax.GradientTransformation | None = None,
    ppo_config: PPOConfig = PPOConfig(),
    checkpoint_path: pathlib.Path | None = None,
    extra_illegal_fn: ExtraIllegalFn | None = None,
):
    """Runs n_iterations of buffered_train_step. Returns (final_variables,
    losses). Defaults (n_episodes=10, ppo_epochs=10, batch_size=64)
    reproduce MaskPlace's own PPO2.py values."""
    if optimizer is None:
        optimizer = optax.adam(learning_rate)
    variables, opt_state, running_stats, key, start_iteration = open_train_state(
        variables, key, optimizer, checkpoint_path
    )

    losses = []
    for i in range(n_iterations):
        key, step_key = make_step_input(key)
        variables, opt_state, running_stats, loss = buffered_train_step(
            step_key, variables, opt_state, running_stats, optimizer, policy_apply_fn, params,
            reward_fn, sizes_array, cell_size, n_episodes, ppo_epochs, batch_size, state_fn, ppo_config,
            extra_illegal_fn,
        )
        losses.append(float(loss))
        checkpoint_every_n(checkpoint_path, 1, start_iteration + i + 1, variables, opt_state, running_stats, key)

    return variables, losses
