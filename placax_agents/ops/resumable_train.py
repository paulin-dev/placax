"""Resumable training: checkpoints the full training state periodically,
evaluates real HPWL periodically, and appends a JSONL log - so a long
run can be interrupted and continued later without losing progress.

checkpoint_every/eval_every count in ABSOLUTE iterations (not relative
to this call), so behavior is identical whether a run happens in one
call or several resumed ones. Supports both sequential (n_envs=1) and
parallel (n_envs>1), dispatched like run.train()."""
import json
import pathlib

from placax._device import recommended_parallelism_mode  # noqa: F401  must precede jax imports
from placax.types import EnvParams, RewardFn  # noqa: F401
from placax_agents.ops.evaluate import evaluate  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.training.algorithm.config import PPOConfig  # noqa: F401
from placax_agents.training.loops.common import (  # noqa: F401
    checkpoint_every_n,
    make_step_input,
    open_train_state,
    save_train_state,
)
from placax_agents.training.loops.parallel_train import _jitted_parallel_train_step
from placax_agents.training.loops.train import _jitted_train_step
from placax_agents.types import AlgorithmFn, StateFn  # noqa: F401

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
    """Real HPWL at current_iteration, or None on non-evaluated iterations."""
    if current_iteration % eval_every != 0:
        return None
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
    log_path: pathlib.Path | None = None,
    n_envs: int = 1,
    mode: str | None = None,
    snapshot_dir: pathlib.Path | None = None,
    snapshot_every: int | None = None,
):
    """Runs n_iterations MORE iterations, resuming from checkpoint_path
    if it exists. Returns (final_variables, log) - log is a list of
    {iteration, loss, real_hpwl} dicts for iterations run in this call
    (real_hpwl is None on non-evaluated iterations); if log_path is
    given each entry is also appended there as a JSON line, so full
    history survives across resumed calls.

    snapshot_dir/snapshot_every, if both given, additionally save a
    permanent, never-overwritten copy at checkpoint_iter_{N}.bin every
    snapshot_every iterations - checkpoint_path itself is always the
    overwritten "resume from here" file."""
    if optimizer is None:
        optimizer = optax.adam(learning_rate)
    use_parallel = recommended_parallelism_mode(mode) == "parallel" and n_envs > 1
    jitted_step = _jitted_parallel_train_step if use_parallel else _jitted_train_step

    variables, opt_state, running_stats, key, start_iteration = open_train_state(
        variables, key, optimizer, checkpoint_path
    )
    if snapshot_dir is not None and snapshot_every is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    log = []
    for i in range(n_iterations):
        current_iteration = start_iteration + i + 1

        key, step_input = make_step_input(key, n_envs if use_parallel else None)
        variables, opt_state, running_stats, loss, _ = jitted_step(
            step_input, variables, opt_state, running_stats, optimizer, policy_apply_fn,
            params, reward_fn, sizes_array, cell_size, state_fn, ppo_config,
        )

        real_hpwl = _maybe_evaluate(
            current_iteration, eval_every, variables, policy_apply_fn, params, sizes_array, cell_size,
            padded_pin_idx, padded_pin_offset, valid_mask, state_fn,
        )
        _append_log_entry(log, log_path, current_iteration, loss, real_hpwl)

        checkpoint_every_n(
            checkpoint_path, checkpoint_every, current_iteration, variables, opt_state, running_stats, key
        )
        snapshot_path = snapshot_dir / f"checkpoint_iter_{current_iteration}.bin" if snapshot_dir else None
        checkpoint_every_n(snapshot_path, snapshot_every, current_iteration, variables, opt_state, running_stats, key)

    save_train_state(checkpoint_path, variables, opt_state, running_stats, key, start_iteration + n_iterations)
    return variables, log
