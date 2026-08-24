"""Sequential training: one episode, one gradient update, repeated.
See parallel_train.py for the batched version - the rollout/loss part
genuinely differs (vmapped vs not, a measured ~28% overhead even at
n_envs=1); the optimizer tail is shared via optimizer_step.py, and
checkpoint state handling via common.py."""
import pathlib

from placax.types import EnvParams, RewardFn  # noqa: F401  must precede jax imports
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.training.algorithm.config import PPOConfig  # noqa: F401
from placax_agents.training.algorithm.gae import compute_gae  # noqa: F401
from placax_agents.training.algorithm.loss import ppo_loss  # noqa: F401
from placax_agents.training.algorithm.optimizer_step import apply_gradient_update  # noqa: F401
from placax_agents.training.algorithm.running_stats import RunningStats  # noqa: F401
from placax_agents.training.loops.common import open_train_state, save_train_state
from placax_agents.training.rollout import collect_rollout  # noqa: F401
from placax_agents.types import AlgorithmFn, StateFn  # noqa: F401

import jax
import jax.numpy as jnp
import optax


def train_step(
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
    state_fn: StateFn = observation,
    ppo_config: PPOConfig = PPOConfig(),
):
    """One full episode + one gradient update. Returns (variables,
    opt_state, running_stats, loss, final_state)."""
    trajectory, final_state = collect_rollout(
        key, variables, policy_apply_fn, params, reward_fn, sizes_array, cell_size, state_fn
    )
    advantages, returns = compute_gae(
        trajectory["reward"], trajectory["value"], trajectory["done"], next_value=jnp.array(0.0),
        gamma=ppo_config.gamma, lam=ppo_config.lam,
    )

    def loss_fn(policy_params, normalized_advantages, normalized_returns):
        return ppo_loss(
            policy_params, policy_apply_fn, trajectory, normalized_advantages, normalized_returns,
            sizes_array, cell_size, params,
            clip_eps=ppo_config.clip_eps, value_coef=ppo_config.value_coef,
            entropy_coef=ppo_config.entropy_coef,
        )

    new_variables, new_opt_state, new_running_stats, loss = apply_gradient_update(
        variables, opt_state, running_stats, optimizer, loss_fn, advantages, returns
    )
    return new_variables, new_opt_state, new_running_stats, loss, final_state


# Built once at import; each distinct jax.jit wrapper would otherwise pay
# its own one-time compile cost on first use.
_jitted_train_step = jax.jit(
    train_step,
    static_argnames=("optimizer", "policy_apply_fn", "reward_fn", "state_fn", "ppo_config"),
)


def train_sequential(
    key: jax.Array,
    variables,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    reward_fn: RewardFn,
    sizes_array: jax.Array,
    cell_size: float,
    n_iterations: int,
    learning_rate: float = 3e-4,
    state_fn: StateFn = observation,
    optimizer: optax.GradientTransformation | None = None,
    ppo_config: PPOConfig = PPOConfig(),
    checkpoint_path: pathlib.Path | None = None,
):
    """Runs n_iterations of train_step. Returns (final_variables, losses).
    The underlying implementation behind run.train(); call directly to
    force sequential. optimizer defaults to optax.adam(learning_rate).
    checkpoint_path, if given, saves the full training state after every
    iteration and auto-resumes from it if it already exists; for periodic
    checkpointing and evaluation snapshots use ops.resumable_train."""
    if optimizer is None:
        optimizer = optax.adam(learning_rate)
    variables, opt_state, running_stats, key, start_iteration = open_train_state(
        variables, key, optimizer, checkpoint_path
    )

    losses = []
    for i in range(n_iterations):
        key, step_key = jax.random.split(key)
        variables, opt_state, running_stats, loss, _ = _jitted_train_step(
            step_key, variables, opt_state, running_stats, optimizer, policy_apply_fn,
            params, reward_fn, sizes_array, cell_size, state_fn, ppo_config,
        )
        losses.append(float(loss))
        if checkpoint_path is not None:
            save_train_state(
                checkpoint_path, variables, opt_state, running_stats, key, start_iteration + i + 1
            )

    return variables, losses
