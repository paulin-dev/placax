"""Save/load Flax variables to/from disk - the foundation both local
checkpoint reuse and pretrained-weight loading need: loading pretrained
weights from a URL is just "download a checkpoint file, then load it
the same way as any local one," not a separate mechanism."""
import pathlib
import urllib.request

from flax import serialization


def save_checkpoint(variables, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(variables))


def load_checkpoint(variables_template, path: pathlib.Path):
    """variables_template: a variables pytree with the same structure as
    what was saved (e.g. straight from policy.init(...)) - flax's
    serialization deserializes into an existing structure, since the
    saved bytes alone don't carry shape/dtype information."""
    return serialization.from_bytes(variables_template, path.read_bytes())


def load_pretrained_from_url(variables_template, url: str, cache_path: pathlib.Path):
    """Downloads a checkpoint from url if not already cached at
    cache_path, then loads it via load_checkpoint - not a separate
    mechanism, just a download step in front of the same local load."""
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)
    return load_checkpoint(variables_template, cache_path)
