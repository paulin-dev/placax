from placax import _device  # noqa: F401  must precede jax imports
from placax_agents.ops.checkpoint import save_checkpoint
from placax_agents.ops.inference import is_bare_checkpoint, positions_to_named_lower_left

import jax.numpy as jnp
import pytest


def test_is_bare_checkpoint_true_for_variables_and_real_hpwl(tmp_path) -> None:
    path = tmp_path / "anything.bin"  # deliberately not named best_checkpoint.bin
    save_checkpoint({"variables": {"params": {}}, "real_hpwl": jnp.array(1.0)}, path)
    assert is_bare_checkpoint(path) is True


def test_is_bare_checkpoint_false_for_full_training_state(tmp_path) -> None:
    path = tmp_path / "checkpoint.bin"
    save_checkpoint(
        {
            "variables": {"params": {}}, "opt_state": (), "running_stats": {}, "iteration": jnp.array(3),
            "key": jnp.array([0, 0], dtype=jnp.uint32),
        },
        path,
    )
    assert is_bare_checkpoint(path) is False


def test_is_bare_checkpoint_rejects_unrecognized_shape(tmp_path) -> None:
    path = tmp_path / "weird.bin"
    save_checkpoint({"something_else": 1}, path)
    with pytest.raises(ValueError, match="unrecognized top-level keys"):
        is_bare_checkpoint(path)


def test_positions_to_named_lower_left_matches_name_to_idx() -> None:
    positions = jnp.array([[0, 0], [2, 3]])
    sizes_array = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    result = positions_to_named_lower_left(positions, sizes_array, cell_size=10.0, name_to_idx={"a": 0, "b": 1})
    assert result == {"a": (0, 0), "b": (20, 30)}
