"""Running mean/variance across many episodes, via Chan's parallel
algorithm (the standard way to combine batch statistics incrementally
without storing every past value)."""
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RunningStats:
    mean: jax.Array
    var: jax.Array
    count: jax.Array


def init_running_stats() -> RunningStats:
    # count starts near-zero, not exactly zero, to avoid a div-by-zero
    # on the very first update.
    return RunningStats(mean=jnp.array(0.0), var=jnp.array(1.0), count=jnp.array(1e-4))


def update_running_stats(stats: RunningStats, x: jax.Array) -> RunningStats:
    """Combines stats with a new batch x (any shape, flattened)."""
    x = x.ravel()
    batch_mean, batch_var, batch_count = x.mean(), x.var(), x.size

    delta = batch_mean - stats.mean
    total_count = stats.count + batch_count

    new_mean = stats.mean + delta * batch_count / total_count
    m_a = stats.var * stats.count
    m_b = batch_var * batch_count
    combined_m2 = m_a + m_b + delta**2 * stats.count * batch_count / total_count
    new_var = combined_m2 / total_count

    return RunningStats(mean=new_mean, var=new_var, count=total_count)


def normalize_with_stats(stats: RunningStats, x: jax.Array, eps: float = 1e-8) -> jax.Array:
    return (x - stats.mean) / jnp.sqrt(stats.var + eps)
