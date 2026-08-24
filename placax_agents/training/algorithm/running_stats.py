"""Running mean/variance across episodes via Chan's parallel algorithm
(the standard way to merge batch statistics incrementally)."""
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RunningStats:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def init_running_stats() -> RunningStats:
    # count starts near-zero, not exactly zero, to avoid div-by-zero on
    # the very first update.
    return RunningStats(mean=jnp.array(0.0), var=jnp.array(1.0), count=jnp.array(1e-4))


def update_running_stats(stats: RunningStats, x: jnp.ndarray) -> RunningStats:
    """Merges a new batch x (any shape, flattened) into the stats."""
    x = x.ravel()
    batch_mean, batch_var, batch_count = x.mean(), x.var(), x.size

    delta = batch_mean - stats.mean
    total_count = stats.count + batch_count

    new_mean = stats.mean + delta * batch_count / total_count
    combined_m2 = (
        stats.var * stats.count + batch_var * batch_count + delta**2 * stats.count * batch_count / total_count
    )
    return RunningStats(mean=new_mean, var=combined_m2 / total_count, count=total_count)


def normalize_with_stats(stats: RunningStats, x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Standardizes x with the running mean/std."""
    return (x - stats.mean) / jnp.sqrt(stats.var + eps)
