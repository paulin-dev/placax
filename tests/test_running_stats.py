from placax_agents.training.algorithm.running_stats import init_running_stats, normalize_with_stats  # noqa: F401
from placax_agents.training.algorithm.running_stats import update_running_stats  # noqa: F401  must precede jax imports

import numpy as np
import jax.numpy as jnp


def test_init_running_stats_is_explicitly_float32() -> None:
    # Must not silently inherit JAX_ENABLE_X64's default-dtype side effect (placax/_device.py) -
    # a weak-typed float64 default here is what let a resumed checkpoint's genuinely-float32 stats
    # (see update_running_stats, which absorbs the netlist's deliberately-float32 sizes_array/
    # padded_pin_offset dtype) collide into a jax.lax.scan carry-dtype mismatch on resume.
    stats = init_running_stats()
    assert stats.mean.dtype == jnp.float32
    assert stats.var.dtype == jnp.float32
    assert stats.count.dtype == jnp.float32


def test_update_running_stats_stays_float32_with_a_float32_batch() -> None:
    stats = init_running_stats()
    updated = update_running_stats(stats, jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32))
    assert updated.mean.dtype == jnp.float32
    assert updated.var.dtype == jnp.float32


def test_running_stats_matches_numpy_on_concatenated_batches() -> None:
    batch1 = np.array([1.0, 2.0, 3.0])
    batch2 = np.array([10.0, 20.0, 30.0, 40.0])

    stats = init_running_stats()
    stats = update_running_stats(stats, jnp.array(batch1))
    stats = update_running_stats(stats, jnp.array(batch2))

    combined = np.concatenate([batch1, batch2])
    assert abs(float(stats.mean) - combined.mean()) < 1e-2
    assert abs(float(stats.var) - combined.var()) < 1e-1


def test_running_stats_three_batches_still_matches_numpy() -> None:
    batches = [np.array([5.0, 5.0]), np.array([-3.0, 8.0, 1.0]), np.array([100.0])]
    stats = init_running_stats()
    for batch in batches:
        stats = update_running_stats(stats, jnp.array(batch))

    combined = np.concatenate(batches)
    assert abs(float(stats.mean) - combined.mean()) < 1e-2
    assert abs(float(stats.var) - combined.var()) < 1e-1


def test_normalize_with_stats_centers_and_scales() -> None:
    stats = init_running_stats()
    stats = update_running_stats(stats, jnp.array([0.0, 10.0]))  # mean=5, var=25

    result = normalize_with_stats(stats, jnp.array([5.0]))
    assert abs(float(result[0])) < 1e-3  # exactly at the mean -> ~0
