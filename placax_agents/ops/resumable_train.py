"""Training loop with periodic checkpointing, HPWL evaluation, and a
JSONL log - resumable across calls (checkpoint_every/eval_every count
in absolute iterations, not relative to a single call)."""
import json
import pathlib

from placax._device import recommended_parallelism_mode  # must precede jax imports
from placax.log import Log
from placax.types import EnvParams, RewardFn
from placax_agents.ops.evaluate import evaluate
from placax_agents.policy.observation import observation
from placax_agents.training.algorithm.config import PPOConfig
from placax_agents.training.loops.common import (
    checkpoint_every_n,
    make_step_input,
    open_train_state,
    save_train_state,
)
from placax_agents.training.loops.parallel_train import _jitted_parallel_train_step
from placax_agents.training.loops.train import _jitted_train_step
from placax_agents.types import AlgorithmFn, StateFn

import jax
import optax


def _maybe_evaluate(
    current_iteration: int,
    eval_every: int,
    variables,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    sizes_array: jax.Array,
    cell_size: float,
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    state_fn: StateFn,
) -> float | None:
    """Returns real HPWL at current_iteration, or None if this isn't an eval iteration."""
    if current_iteration % eval_every != 0:
        return None
    # Run a full greedy rollout with the current policy just to measure quality, not to train.
    _positions, hpwl_value = evaluate(
        variables, policy_apply_fn, params, sizes_array, cell_size,
        padded_pin_idx, padded_pin_offset, valid_mask, state_fn,
    )
    return float(hpwl_value)


def _append_log_entry(
    log: list[dict], log_path: pathlib.Path | None, iteration: int, loss: float, real_hpwl: float | None
) -> None:
    """Appends {iteration, loss, real_hpwl} to log, and to log_path as a JSON line if given."""
    entry = {"iteration": iteration, "loss": float(loss), "real_hpwl": real_hpwl}
    log.append(entry)
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def resumable_train(
    checkpoint_path: pathlib.Path,
    variables,
    key: jax.Array,
    policy_apply_fn: AlgorithmFn,
    params: EnvParams,
    reward_fn: RewardFn,
    sizes_array: jax.Array,
    cell_size: float,
    n_iterations: int,
    padded_pin_idx: jax.Array,
    padded_pin_offset: jax.Array,
    valid_mask: jax.Array,
    learning_rate: float = 3e-4,
    state_fn: StateFn = observation,
    optimizer: optax.GradientTransformation | None = None,
    ppo_config: PPOConfig = PPOConfig(),
    checkpoint_every: int = 10,
    eval_every: int = 10,
    log_every: int | None = 1,
    log_path: pathlib.Path | None = None,
    n_envs: int = 1,
    mode: str | None = None,
    snapshot_dir: pathlib.Path | None = None,
    snapshot_every: int | None = None,
):
    """Runs n_iterations more of training, resuming from checkpoint_path if it exists, and returns (final_variables, log)."""
    if optimizer is None:
        optimizer = optax.adam(learning_rate)
    # Pick the sequential or parallel training-step implementation based on hardware + n_envs.
    use_parallel = recommended_parallelism_mode(mode) == "parallel" and n_envs > 1
    jitted_step = _jitted_parallel_train_step if use_parallel else _jitted_train_step

    # Resume from a checkpoint if one exists, otherwise start fresh from the given variables/key.
    variables, opt_state, running_stats, key, start_iteration = open_train_state(
        variables, key, optimizer, checkpoint_path
    )
    if snapshot_dir is not None and snapshot_every is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    log = []
    for i in range(n_iterations):
        current_iteration = start_iteration + i + 1

        # 1. one gradient step
        key, step_input = make_step_input(key, n_envs if use_parallel else None)
        variables, opt_state, running_stats, loss, _ = jitted_step(
            step_input, variables, opt_state, running_stats, optimizer, policy_apply_fn,
            params, reward_fn, sizes_array, cell_size, state_fn, ppo_config,
        )

        # 2. periodic real-HPWL eval + 3. log entry (always, eval or not)
        real_hpwl = _maybe_evaluate(
            current_iteration, eval_every, variables, policy_apply_fn, params, sizes_array, cell_size,
            padded_pin_idx, padded_pin_offset, valid_mask, state_fn,
        )
        _append_log_entry(log, log_path, current_iteration, loss, real_hpwl)
        if log_every is not None and current_iteration % log_every == 0:
            hpwl_str = f"{real_hpwl:.1f}" if real_hpwl is not None else "-"
            Log.info(f"iter {current_iteration:>6}  loss={loss:>10.4f}  real_hpwl={hpwl_str}")

        # 4. periodic checkpoint (overwritten) + 5. periodic snapshot (never overwritten)
        checkpoint_every_n(
            checkpoint_path, checkpoint_every, current_iteration, variables, opt_state, running_stats, key
        )
        snapshot_path = snapshot_dir / f"checkpoint_iter_{current_iteration}.bin" if snapshot_dir else None
        checkpoint_every_n(snapshot_path, snapshot_every, current_iteration, variables, opt_state, running_stats, key)

    # Always checkpoint at the end, regardless of checkpoint_every.
    save_train_state(checkpoint_path, variables, opt_state, running_stats, key, start_iteration + n_iterations)
    return variables, log
