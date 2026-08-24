"""Checkpoint plumbing shared by the training loops: one bundle layout
(variables, opt_state, running_stats, iteration, key) saved/resumed by
train_sequential, train_parallel, and resumable_train alike."""
import pathlib

from placax import _device  # noqa: F401  must precede jax imports
from placax_agents.ops.checkpoint import load_checkpoint, save_checkpoint  # noqa: F401
from placax_agents.training.algorithm.running_stats import RunningStats, init_running_stats  # noqa: F401

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
    """Initializes optimizer/running stats fresh, or resumes them from an
    existing checkpoint. Returns (variables, opt_state, running_stats,
    key, start_iteration); variables/key double as the deserialization
    template when loading."""
    template = train_state_bundle(variables, optimizer.init(variables), init_running_stats(), key, 0)
    state = load_checkpoint(template, checkpoint_path) if checkpoint_path is not None and checkpoint_path.exists() else template
    return (
        state["variables"],
        state["opt_state"],
        state["running_stats"],
        state["key"],
        int(state["iteration"]),
    )
