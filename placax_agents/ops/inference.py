"""Loads trained policy weights for pure inference (production placement, never training) and converts a
completed rollout's grid positions back into the named, real-unit coordinates external tools need."""
import pathlib

from placax_agents.ops.checkpoint import load_checkpoint
from placax_agents.policy.scale import to_real_lower_left
from placax_agents.training.loops.common import open_train_state

import jax.numpy as jnp
import numpy as np
from flax import serialization
from jax import random

_BARE_KEYS = {"variables", "real_hpwl"}
_FULL_KEYS = {"variables", "opt_state", "running_stats", "iteration", "key"}


def is_bare_checkpoint(checkpoint_path: pathlib.Path) -> bool:
    """Whether checkpoint_path holds a bare-weights bundle ({"variables", "real_hpwl"}, e.g. what
    _save_best_checkpoint writes) rather than a full training-state bundle ({"variables", "opt_state",
    "running_stats", "iteration", "key"}) - detected from the file's own top-level keys (via
    serialization.msgpack_restore, which needs no template) rather than guessed from its filename, so
    this works regardless of what the file is called (e.g. a renamed/migrated checkpoint)."""
    top_level_keys = set(serialization.msgpack_restore(checkpoint_path.read_bytes()))
    if top_level_keys == _BARE_KEYS:
        return True
    if top_level_keys == _FULL_KEYS:
        return False
    raise ValueError(
        f"{checkpoint_path} has unrecognized top-level keys {sorted(top_level_keys)} - expected either "
        f"{sorted(_BARE_KEYS)} (bare) or {sorted(_FULL_KEYS)} (full training state)"
    )


def load_policy_variables(variables_template, checkpoint_path: pathlib.Path, bare: bool, optimizer=None):
    """bare=True reads a best_checkpoint.bin-shaped bundle ({"variables", "real_hpwl"}) - the production
    default, no optimizer/RNG state involved at all. bare=False reads a full checkpoint.bin training-state
    bundle: it still needs the SAME optimizer the run was trained with, only to match the saved opt_state's
    pytree shape for deserialization - training never happens here either way, and opt_state is discarded
    immediately after loading."""
    if bare:
        template = {"variables": variables_template, "real_hpwl": jnp.array(0.0)}
        return load_checkpoint(template, checkpoint_path)["variables"]
    if optimizer is None:
        raise ValueError("bare=False (a full checkpoint.bin) needs the optimizer it was trained with")
    variables, _opt_state, _running_stats, _key, _iteration = open_train_state(
        variables_template, random.PRNGKey(0), optimizer, checkpoint_path
    )
    return variables


def positions_to_named_lower_left(
    positions, sizes_array, cell_size: float, name_to_idx: dict[str, int]
) -> dict[str, tuple[int, int]]:
    """Converts a completed rollout's grid positions into {macro_name: (x, y)} real-unit lower-left
    corners, integer-rounded for Bookshelf .pl / DEF PLACED - the shape write_pl/write_placed_def need."""
    real_lower_left = np.asarray(to_real_lower_left(positions, cell_size))
    return {
        name: (int(round(real_lower_left[idx, 0])), int(round(real_lower_left[idx, 1])))
        for name, idx in name_to_idx.items()
    }
