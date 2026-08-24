"""Save/load Flax variables to/from disk - the foundation both local
checkpoint reuse and pretrained-weight loading share."""
import pathlib
import urllib.request

from flax import serialization


def save_checkpoint(variables, path: pathlib.Path) -> None:
    """Serializes variables to path (creating parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(variables))


def load_checkpoint(variables_template, path: pathlib.Path):
    """Deserializes into an existing pytree structure (the saved bytes
    carry no shape/dtype info - the template supplies it)."""
    return serialization.from_bytes(variables_template, path.read_bytes())


def load_pretrained_from_url(variables_template, url: str, cache_path: pathlib.Path):
    """Downloads the checkpoint to cache_path if absent, then loads it
    like any local one."""
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)
    return load_checkpoint(variables_template, cache_path)
