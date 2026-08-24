"""Resumable training: checkpoints the full training state (weights,
optimizer state, running stats, iteration count, PRNG key) periodically,
and evaluates real HPWL periodically too - so a long run can be
interrupted and continued later, locally or on different (e.g. GPU)
hardware, without losing progress or restarting from scratch.

Builds on train.py's _train_n_steps and parallel_train.py's
_train_n_parallel_steps, not its own copy of either loop: an earlier
version reimplemented train_sequential's loop, which is exactly the
kind of duplication that drifts out of sync over time. train_sequential/
train_parallel discard opt_state/running_stats at the end for a simple
public API; this module is the one place that actually needs them
preserved, so it calls the shared primitives directly instead.

Supports both sequential (n_envs=1, the default) and parallel
(n_envs>1) - added once a real need for parallel resumable training
came up, not built speculatively ahead of that."""
import json
import pathlib

from placax._device import recommended_parallelism_mode  # noqa: F401  must precede jax imports
from placax.types import EnvParams, RewardFn  # noqa: F401
from placax_agents.ops.checkpoint import load_checkpoint, save_checkpoint  # noqa: F401
from placax_agents.ops.evaluate import evaluate  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.training.algorithm.config import PPOConfig  # noqa: F401
from placax_agents.training.algorithm.running_stats import init_running_stats  # noqa: F401
from placax_agents.training.loops.parallel_train import (  # noqa: F401
    _build_jitted_parallel_train_step, _train_n_parallel_steps,
)
from placax_agents.training.loops.train import _build_jitted_train_step, _train_n_steps  # noqa: F401
from placax_agents.types import AlgorithmFn, StateFn  # noqa: F401

import jax
import jax.numpy as jnp
import optax


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
    if it already exists (using variables/key only as the fresh-start
    values and the template shape for loading), or starting fresh
    otherwise. Returns (final_variables, log) - log is a list of
    {iteration, loss, real_hpwl} dicts for iterations run in this call
    (real_hpwl is None on iterations that weren't evaluated).

    checkpoint_every/eval_every count in absolute iterations, not
    relative to this call - so behavior is identical whether a run
    happens in one call or several resumed ones. If log_path is given,
    each entry is also appended there as a JSON line, so a full history
    survives across resumed calls even though the returned log only
    covers this one.

    n_envs/mode: same as train()'s own dispatch (run.py) - mode=None
    auto-detects sequential vs parallel via recommended_parallelism_mode(),
    or force one explicitly. n_envs only matters when parallel is used.

    snapshot_dir/snapshot_every: if both given, ALSO saves a permanent,
    never-overwritten copy at checkpoint_iter_{N}.bin every
    snapshot_every iterations - checkpoint_path itself is always
    overwritten (it's "resume from here"), so without a separate
    snapshot you can't roll back to an earlier point once training has
    moved past it. Snapshots are named by iteration number, not
    datetime: iteration count is the meaningful clock for resumable
    training (wall-clock time between runs is arbitrary), and the
    filesystem's own modification time already gives you a real
    timestamp for free if you ever need one, without encoding it
    redundantly into the filename. Checkpoints are small (~32KB for
    this policy size, confirmed directly) - keeping many is cheap."""
    if optimizer is None:
        optimizer = optax.adam(learning_rate)
    resolved_mode = recommended_parallelism_mode(mode)
    use_parallel = resolved_mode == "parallel" and n_envs > 1

    template = {
        "variables": variables,
        "opt_state": optimizer.init(variables),
        "running_stats": init_running_stats(),
        "iteration": jnp.array(0),
        "key": key,
    }

    if checkpoint_path.exists():
        state = load_checkpoint(template, checkpoint_path)
    else:
        state = template

    variables = state["variables"]
    opt_state = state["opt_state"]
    running_stats = state["running_stats"]
    key = state["key"]
    start_iteration = int(state["iteration"])

    if use_parallel:
        jitted_step = _build_jitted_parallel_train_step()
    else:
        jitted_step = _build_jitted_train_step()

    log = []
    for i in range(n_iterations):
        current_iteration = start_iteration + i + 1

        if use_parallel:
            variables, opt_state, running_stats, key, step_losses = _train_n_parallel_steps(
                jitted_step, key, variables, opt_state, running_stats, optimizer,
                policy_apply_fn, params, reward_fn, sizes_array, cell_size, n_envs, 1,
                state_fn, ppo_config,
            )
        else:
            variables, opt_state, running_stats, key, step_losses = _train_n_steps(
                jitted_step, key, variables, opt_state, running_stats, optimizer,
                policy_apply_fn, params, reward_fn, sizes_array, cell_size, 1,
                state_fn, ppo_config,
            )
        loss = step_losses[0]

        real_hpwl = None
        if current_iteration % eval_every == 0:
            _positions, hpwl_value = evaluate(
                variables, policy_apply_fn, params, sizes_array, cell_size,
                padded_pin_idx, padded_pin_offset, valid_mask, state_fn,
            )
            real_hpwl = float(hpwl_value)

        entry = {"iteration": current_iteration, "loss": loss, "real_hpwl": real_hpwl}
        log.append(entry)
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

        if current_iteration % checkpoint_every == 0:
            bundle = {
                "variables": variables, "opt_state": opt_state, "running_stats": running_stats,
                "iteration": jnp.array(current_iteration), "key": key,
            }
            save_checkpoint(bundle, checkpoint_path)

        if snapshot_dir is not None and snapshot_every is not None:
            if current_iteration % snapshot_every == 0:
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                bundle = {
                    "variables": variables, "opt_state": opt_state,
                    "running_stats": running_stats,
                    "iteration": jnp.array(current_iteration), "key": key,
                }
                save_checkpoint(bundle, snapshot_dir / f"checkpoint_iter_{current_iteration}.bin")

    final_bundle = {
        "variables": variables, "opt_state": opt_state, "running_stats": running_stats,
        "iteration": jnp.array(start_iteration + n_iterations), "key": key,
    }
    save_checkpoint(final_bundle, checkpoint_path)

    return variables, log
