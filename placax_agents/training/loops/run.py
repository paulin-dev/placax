"""The single entry point for training - dispatches to sequential or
parallel based on recommended_parallelism_mode() (auto-detected from
JAX backend, or overridden manually), rather than making every caller
choose between two separate functions themselves. train_sequential and
train_parallel (in train.py and parallel_train.py) remain directly
callable if you want to force one specifically."""
import pathlib

from placax._device import recommended_parallelism_mode  # noqa: F401  must precede jax imports
from placax.types import EnvParams, RewardFn  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.training.algorithm.config import PPOConfig  # noqa: F401
from placax_agents.training.loops.parallel_train import train_parallel  # noqa: F401
from placax_agents.training.loops.train import train_sequential  # noqa: F401
from placax_agents.types import AlgorithmFn, StateFn  # noqa: F401

import jax
import optax


def train(
    key: jax.Array,
    variables,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    reward_fn: RewardFn,
    sizes_array: jax.Array,
    cell_size: float,
    n_iterations: int,
    n_envs: int = 1,
    mode: str | None = None,
    learning_rate: float = 3e-4,
    state_fn: StateFn = observation,
    optimizer: optax.GradientTransformation | None = None,
    ppo_config: PPOConfig = PPOConfig(),
    checkpoint_path: pathlib.Path | None = None,
):
    """Runs n_iterations of training, sequential or parallel depending
    on `mode`:
        mode=None (default): auto-detected via recommended_parallelism_mode()
        mode="sequential" or "parallel": forced explicitly
    n_envs only matters when the resolved mode is "parallel" - how many
    episodes get collected at once per iteration. Returns
    (final_variables, losses), same shape either way.

    state_fn defaults to observation() - the same swappable axis as
    reward_fn/policy_apply_fn, threaded all the way through. optimizer
    defaults to optax.adam(learning_rate) if not given - any
    optax.GradientTransformation works. ppo_config bundles
    gamma/lam/clip_eps/value_coef/entropy_coef, also threaded all the
    way through to compute_gae/ppo_loss.

    checkpoint_path, if given, saves progress (weights, optimizer
    state, running stats, PRNG key) after every iteration, and resumes
    from it automatically next time you call train() with the same
    path - "save every lap". For periodic (not every-iteration)
    checkpointing, real-HPWL evaluation snapshots over the course of
    training, or a persisted log, use ops.resumable_train instead."""
    resolved_mode = recommended_parallelism_mode(mode)

    if resolved_mode == "sequential" or n_envs <= 1:
        return train_sequential(
            key, variables, policy_apply_fn, params, reward_fn, sizes_array, cell_size,
            n_iterations, learning_rate, state_fn, optimizer, ppo_config, checkpoint_path,
        )
    return train_parallel(
        key, variables, policy_apply_fn, params, reward_fn, sizes_array, cell_size,
        n_envs, n_iterations, learning_rate, state_fn, optimizer, ppo_config, checkpoint_path,
    )
