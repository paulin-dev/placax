"""Entry point for training: dispatches to sequential or parallel based
on recommended_parallelism_mode()."""
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
    """Runs n_iterations of training (mode=None auto-detects). Returns
    (final_variables, losses). n_envs only matters if mode is parallel."""
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
