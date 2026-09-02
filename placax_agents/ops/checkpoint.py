"""Save/load Flax variables to/from disk."""
import pathlib
import urllib.request

from flax import serialization

import jax


def save_checkpoint(variables, path: pathlib.Path) -> None:
    """Serializes variables to path (creating parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(variables))


def load_checkpoint(variables_template, path: pathlib.Path):
    """Deserializes into variables_template (supplies shape/dtype)."""
    restored = serialization.from_bytes(variables_template, path.read_bytes())
    # flax's from_bytes preserves each leaf's ON-DISK dtype rather than the template's (confirmed
    # empirically - it does NOT honor this function's own "supplies shape/dtype" contract on its
    # own). Cast back to the template's dtype so a checkpoint saved under a different dtype regime
    # (e.g. JAX_ENABLE_X64 toggled between the save and this load - see placax/_device.py) can't
    # silently resume with a leaf dtype that no longer matches what the rest of the process computes,
    # which surfaces far downstream as an opaque jax.lax.scan carry-type mismatch.
    return jax.tree_util.tree_map(
        lambda template_leaf, restored_leaf: restored_leaf.astype(template_leaf.dtype),
        variables_template, restored,
    )


def load_pretrained_from_url(variables_template, url: str, cache_path: pathlib.Path):
    """Downloads the checkpoint to cache_path if absent, then loads it like any local one."""
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)
    return load_checkpoint(variables_template, cache_path)
