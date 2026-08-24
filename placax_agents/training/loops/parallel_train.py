"""Parallel training: n_envs episodes collected and averaged into one
gradient update per iteration, via vmap. See train.py for the sequential
version - the rollout/loss computation genuinely differs (vmapped or
not - a real, measured ~28% overhead even at n_envs=1, not just
historical duplication): whether that's worth it depends entirely on
hardware, measured directly at ~73x SLOWER on CPU-only hardware and
~3.8x FASTER on a real GPU (see scripts/compare_sequential_vs_parallel.py).
The optimizer-update tail is identical either way and shared via
optimizer_step.py. This module makes parallel collection possible;
autotune.find_max_batch_size decides whether and how much to use it."""
import pathlib

from placax.types import EnvParams, RewardFn  # noqa: F401  must precede jax imports
from placax_agents.ops.checkpoint import load_checkpoint, save_checkpoint  # noqa: F401
from placax_agents.training.algorithm.config import PPOConfig  # noqa: F401
from placax_agents.training.algorithm.gae import compute_gae  # noqa: F401
from placax_agents.training.algorithm.loss import ppo_loss  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.training.algorithm.optimizer_step import apply_gradient_update  # noqa: F401
from placax_agents.training.rollout import collect_rollout  # noqa: F401
from placax_agents.training.algorithm.running_stats import RunningStats, init_running_stats  # noqa: F401
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
    """Same as train.train_step, but keys carries a leading n_envs
    dimension: n_envs independent episodes are collected and averaged
    into one gradient update, not n_envs separate updates."""
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


def _build_jitted_parallel_train_step():
    """Builds the jitted parallel_train_step function - callers should
    build this once and reuse it, matching train.py's
    _build_jitted_train_step for the same reason (each distinct
    jax.jit(...) wrapper pays its own one-time compilation cost)."""
    return jax.jit(
        parallel_train_step,
        static_argnames=("optimizer", "policy_apply_fn", "reward_fn", "state_fn", "ppo_config"),
    )


def _train_n_parallel_steps(
    jitted_parallel_train_step,
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
    n_envs: int,
    n_iterations: int,
    state_fn: StateFn,
    ppo_config: PPOConfig,
):
    """Runs n_iterations of an ALREADY-BUILT jitted_parallel_train_step,
    each iteration collecting n_envs episodes at once - the parallel
    counterpart of train.py's _train_n_steps, with the same shared-
    primitive shape: (variables, opt_state, running_stats, key, losses)
    in, same out, so train_parallel (simple public API) and
    ops.resumable_train (which needs opt_state/running_stats preserved
    to actually resume) both build on this instead of duplicating it."""
    losses = []
    for _ in range(n_iterations):
        key, subkey = jax.random.split(key)
        step_keys = jax.random.split(subkey, n_envs)
        variables, opt_state, running_stats, loss, _final_states = jitted_parallel_train_step(
            step_keys, variables, opt_state, running_stats, optimizer, policy_apply_fn,
            params, reward_fn, sizes_array, cell_size, state_fn, ppo_config,
        )
        losses.append(float(loss))

    return variables, opt_state, running_stats, key, losses


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
    episodes at once. Returns (final_variables, losses). Called
    train_parallel, not parallel_train: train() itself (run.py) is the
    unified entry point that dispatches between this and
    train.train_sequential - this is the underlying implementation,
    still callable directly if you want to force parallel specifically.

    optimizer defaults to optax.adam(learning_rate) if not given - same
    swappable pattern as train_sequential. ppo_config bundles
    gamma/lam/clip_eps/value_coef/entropy_coef - same reachability fix
    as train_sequential. checkpoint_path, if given, saves the full
    training state after every iteration and resumes automatically if
    it already exists - same simple pattern as train_sequential."""
    if optimizer is None:
        optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(variables)
    running_stats = init_running_stats()
    start_iteration = 0

    if checkpoint_path is not None:
        template = {
            "variables": variables, "opt_state": opt_state, "running_stats": running_stats,
            "iteration": jnp.array(0), "key": key,
        }
        if checkpoint_path.exists():
            state = load_checkpoint(template, checkpoint_path)
            variables, opt_state, running_stats, key = (
                state["variables"], state["opt_state"], state["running_stats"], state["key"]
            )
            start_iteration = int(state["iteration"])

    jitted_parallel_train_step = _build_jitted_parallel_train_step()
    losses = []
    for i in range(n_iterations):
        variables, opt_state, running_stats, key, step_losses = _train_n_parallel_steps(
            jitted_parallel_train_step, key, variables, opt_state, running_stats, optimizer,
            policy_apply_fn, params, reward_fn, sizes_array, cell_size, n_envs, 1, state_fn,
            ppo_config,
        )
        losses.append(step_losses[0])

        if checkpoint_path is not None:
            bundle = {
                "variables": variables, "opt_state": opt_state, "running_stats": running_stats,
                "iteration": jnp.array(start_iteration + i + 1), "key": key,
            }
            save_checkpoint(bundle, checkpoint_path)

    return variables, losses
