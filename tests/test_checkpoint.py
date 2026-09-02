import functools
import http.server
import pathlib
import threading

from placax_agents.ops.checkpoint import load_checkpoint, load_pretrained_from_url, save_checkpoint  # noqa: F401  must precede jax imports
from placax_agents.policy.architectures.cnn import CNNActorCritic  # noqa: F401

import jax
import jax.numpy as jnp
from jax import random


def _all_match(pytree_a, pytree_b) -> bool:
    leaves_a = jax.tree_util.tree_leaves(pytree_a)
    leaves_b = jax.tree_util.tree_leaves(pytree_b)
    return all((a == b).all() for a, b in zip(leaves_a, leaves_b))


def test_save_and_load_checkpoint_round_trip(tmp_path: pathlib.Path) -> None:
    policy = CNNActorCritic()
    variables = policy.init(random.PRNGKey(0), {"canvas": jnp.zeros((8, 8), dtype=bool)})
    mutated = jax.tree_util.tree_map(lambda x: x + 1.0, variables)

    path = tmp_path / "checkpoint.bin"
    save_checkpoint(mutated, path)

    fresh_template = policy.init(random.PRNGKey(999), {"canvas": jnp.zeros((8, 8), dtype=bool)})
    loaded = load_checkpoint(fresh_template, path)

    assert _all_match(mutated, loaded)
    assert not _all_match(fresh_template, loaded)  # confirms it's not just returning the template


def test_load_checkpoint_casts_to_the_templates_dtype(tmp_path: pathlib.Path) -> None:
    # flax's own from_bytes preserves each leaf's ON-DISK dtype rather than the template's (e.g. a
    # checkpoint saved while JAX_ENABLE_X64 was off, now loaded into a float64 template) - this must
    # not silently resume with a leaf dtype that no longer matches the rest of the process, which
    # otherwise only surfaces far downstream as an opaque jax.lax.scan carry-type mismatch.
    saved = {"mean": jnp.array(0.0, dtype=jnp.float32)}
    path = tmp_path / "checkpoint.bin"
    save_checkpoint(saved, path)

    template = {"mean": jnp.array(0.0, dtype=jnp.float64)}
    loaded = load_checkpoint(template, path)

    assert loaded["mean"].dtype == jnp.float64


def test_load_pretrained_from_url_downloads_and_caches(tmp_path: pathlib.Path) -> None:
    policy = CNNActorCritic()
    variables = policy.init(random.PRNGKey(0), {"canvas": jnp.zeros((8, 8), dtype=bool)})

    serve_dir = tmp_path / "server"
    serve_dir.mkdir()
    save_checkpoint(variables, serve_dir / "checkpoint.bin")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(serve_dir))
    server = http.server.HTTPServer(("localhost", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        cache_path = tmp_path / "downloaded.bin"
        template = policy.init(random.PRNGKey(999), {"canvas": jnp.zeros((8, 8), dtype=bool)})

        loaded = load_pretrained_from_url(
            template, f"http://localhost:{port}/checkpoint.bin", cache_path
        )
        assert _all_match(variables, loaded)
        assert cache_path.exists()

        # second call must reuse the cache, not re-download
        cache_mtime = cache_path.stat().st_mtime
        loaded_again = load_pretrained_from_url(
            template, f"http://localhost:{port}/checkpoint.bin", cache_path
        )
        assert _all_match(variables, loaded_again)
        assert cache_path.stat().st_mtime == cache_mtime  # untouched, confirms no re-download
    finally:
        server.shutdown()
