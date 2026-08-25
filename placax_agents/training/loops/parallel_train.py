"""Parallel training: n_envs episodes collected and averaged into one
gradient update per iteration, via vmap. See train.py for sequential."""
import pathlib

from placax.types import EnvParams, RewardFn  # noqa: F401  must precede jax imports
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.training.algorithm.config import PPOConfig  # noqa: F401
from placax_agents.training.algorithm.gae import compute_gae  # noqa: F401
from placax_agents.training.algorithm.loss import ppo_loss  # noqa: F401
from placax_agents.training.algorithm.optimizer_step import apply_gradient_update  # noqa: F401
from placax_agents.training.algorithm.running_stats import RunningStats  # noqa: F401
from placax_agents.training.loops.common import checkpoint_every_n, make_step_input, open_train_state
from placax_agents.training.rollout import collect_rollout  # noqa: F401
from placax_agents.types import AlgorithmFn, StateFn  # noqa: F401

import jax
import jax.numpy as jnp
import optax


def parallel_train_step(
    keys: jax.Array,
    variables,
    opt_state,
    running_stats: RunningStats,
    optimizer: optax.GradientTransformation,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    reward_fn: RewardFn,
    sizes_array: jax.Array,
    cell_size: float,
    state_fn: StateFn = observation,
    ppo_config: PPOConfig = PPOConfig(),
):
    """Like train.train_step, but keys has a leading n_envs dimension:
    n_envs episodes are collected and averaged into one update."""
    # One episode per key, batched via vmap.
    batched_rollout = jax.vmap(
        collect_rollout, in_axes=(0, None, None, None, None, None, None, None)
    )
    trajectories, final_states = batched_rollout(
        keys, variables, policy_apply_fn, params, reward_fn, sizes_array, cell_size, state_fn
    )

    batched_gae = jax.vmap(compute_gae, in_axes=(0, 0, 0, None, None, None))
    advantages, returns = batched_gae(
        trajectories["reward"], trajectories["value"], trajectories["done"], jnp.array(0.0),
        ppo_config.gamma, ppo_config.lam,
    )

    def loss_fn(policy_params, normalized_advantages, normalized_returns):
        # Per-episode PPO loss, averaged into the one gradient update.
        per_episode_losses = jax.vmap(
            lambda traj, adv, ret: ppo_loss(
                policy_params, policy_apply_fn, traj, adv, ret, sizes_array, cell_size, params,
                clip_eps=ppo_config.clip_eps, value_coef=ppo_config.value_coef,
                entropy_coef=ppo_config.entropy_coef,
            )
        )(trajectories, normalized_advantages, normalized_returns)
        return per_episode_losses.mean()

    new_variables, new_opt_state, new_running_stats, loss = apply_gradient_update(
        variables, opt_state, running_stats, optimizer, loss_fn, advantages, returns
    )
    return new_variables, new_opt_state, new_running_stats, loss, final_states


# Built once at import; see train.py for why.
_jitted_parallel_train_step = jax.jit(
    parallel_train_step,
    static_argnames=("optimizer", "policy_apply_fn", "reward_fn", "state_fn", "ppo_config"),
)


def train_parallel(
    key: jax.Array,
    variables,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    reward_fn: RewardFn,
    sizes_array: jax.Array,
    cell_size: float,
    n_envs: int,
    n_iterations: int,
    learning_rate: float = 3e-4,
    state_fn: StateFn = observation,
    optimizer: optax.GradientTransformation | None = None,
    ppo_config: PPOConfig = PPOConfig(),
    checkpoint_path: pathlib.Path | None = None,
):
    """Runs n_iterations of parallel_train_step, each collecting n_envs
    episodes at once. Returns (final_variables, losses)."""
    if optimizer is None:
        optimizer = optax.adam(learning_rate)
    variables, opt_state, running_stats, key, start_iteration = open_train_state(
        variables, key, optimizer, checkpoint_path
    )

    losses = []
    for i in range(n_iterations):
        key, step_keys = make_step_input(key, n_envs)
        variables, opt_state, running_stats, loss, _ = _jitted_parallel_train_step(
            step_keys, variables, opt_state, running_stats, optimizer, policy_apply_fn,
            params, reward_fn, sizes_array, cell_size, state_fn, ppo_config,
        )
        losses.append(float(loss))
        checkpoint_every_n(checkpoint_path, 1, start_iteration + i + 1, variables, opt_state, running_stats, key)

    return variables, losses
