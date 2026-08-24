"""The single entry point for training - dispatches to sequential or
parallel based on recommended_parallelism_mode() (auto-detected from
the JAX backend, or forced via mode=). train_sequential/train_parallel
remain directly callable if you want to force one specifically."""
import pathlib

from placax._device import recommended_parallelism_mode  # noqa: F401  must precede jax imports
from placax.types import EnvParams, RewardFn  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.training.algorithm.config import PPOConfig  # noqa: F401
from placax_agents.training.loops.parallel_train import train_parallel  # noqa: F401
from placax_agents.training.loops.train import train_sequential  # noqa: F401
from placax_agents.types import AlgorithmFn, StateFn  # noqa: F401

import optax


def train(
    key,
    variables,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    reward_fn: RewardFn,
    sizes_array,
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
    """Runs n_iterations of training, sequential or parallel depending on
    mode (None = auto-detect; "sequential"/"parallel" = force). n_envs
    only matters when the resolved mode is parallel. Returns
    (final_variables, losses), same shape either way. optimizer defaults
    to optax.adam(learning_rate). checkpoint_path, if given, saves the
    full training state after every iteration and auto-resumes from it;
    for periodic checkpointing / eval snapshots use ops.resumable_train."""
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
