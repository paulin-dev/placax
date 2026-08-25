"""Checkpoint and per-iteration plumbing shared by the training loops."""
import pathlib

from placax import _device  # noqa: F401  must precede jax imports
from placax_agents.ops.checkpoint import load_checkpoint, save_checkpoint
from placax_agents.training.algorithm.running_stats import RunningStats, init_running_stats

import jax
import jax.numpy as jnp


def train_state_bundle(
    variables, opt_state, running_stats: RunningStats, key: jnp.ndarray, iteration: int
) -> dict:
    """The full resumable training state as a flat dict."""
    return {
        "variables": variables,
        "opt_state": opt_state,
        "running_stats": running_stats,
        "iteration": jnp.array(iteration),
        "key": key,
    }


def save_train_state(path: pathlib.Path, variables, opt_state, running_stats, key, iteration: int) -> None:
    """Saves the full training state to path (overwrites)."""
    save_checkpoint(train_state_bundle(variables, opt_state, running_stats, key, iteration), path)


def open_train_state(variables, key, optimizer, checkpoint_path: pathlib.Path | None):
    """Fresh optimizer/running stats, or resumed from checkpoint_path if
    it exists. Returns (variables, opt_state, running_stats, key,
    start_iteration)."""
    template = train_state_bundle(variables, optimizer.init(variables), init_running_stats(), key, 0)
    # template also serves as the deserialization target - a checkpoint is just bytes,
    # it needs a matching pytree structure to know how to unpack itself.
    state = load_checkpoint(template, checkpoint_path) if checkpoint_path is not None and checkpoint_path.exists() else template
    return (
        state["variables"],
        state["opt_state"],
        state["running_stats"],
        state["key"],
        int(state["iteration"]),
    )


def make_step_input(key: jax.Array, n_envs: int | None = None) -> tuple[jax.Array, jax.Array]:
    """(key, step_input) for one iteration: n_envs=None gives a single
    subkey (sequential); n_envs=N gives N split keys (parallel)."""
    key, step_key = jax.random.split(key)
    if n_envs is None:
        return key, step_key
    return key, jax.random.split(step_key, n_envs)


def checkpoint_every_n(
    path: pathlib.Path | None,
    every: int | None,
    iteration: int,
    variables,
    opt_state,
    running_stats: RunningStats,
    key: jax.Array,
) -> None:
    """Saves to path if iteration % every == 0. No-op if path or every is None."""
    if path is not None and every is not None and iteration % every == 0:
        save_train_state(path, variables, opt_state, running_stats, key, iteration)
