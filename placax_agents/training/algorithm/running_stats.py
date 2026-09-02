"""Running mean/variance across episodes."""
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RunningStats:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def init_running_stats() -> RunningStats:
    # count starts near-zero, not exactly zero, to avoid div-by-zero on the very first update.
    # Explicit float32, not left to inherit whatever JAX_ENABLE_X64 (placax/_device.py) happens to
    # default untyped literals to: this matches both what these values naturally settle into during
    # real training anyway (the netlist's sizes_array/padded_pin_offset are deliberately float32 -
    # placax/netlist/padding.py - and that propagates through returns into update_running_stats
    # below) and MaskPlace's own PyTorch reference, which never asks for double precision on this
    # quantity - only its masked-logit-before-softmax upcast does. Being explicit also means this no
    # longer silently rides on x64's global default-dtype side effect: an implicitly-typed (weak)
    # float64 default here is what let a resumed run's checkpoint - genuinely float32 on disk, same
    # reason - collide into an opaque jax.lax.scan carry-dtype mismatch (see
    # placax_agents/ops/checkpoint.py::load_checkpoint's own dtype-cast fix, kept regardless as a
    # general safety net for any future dtype-regime drift).
    return RunningStats(
        mean=jnp.array(0.0, dtype=jnp.float32),
        var=jnp.array(1.0, dtype=jnp.float32),
        count=jnp.array(1e-4, dtype=jnp.float32),
    )


def update_running_stats(stats: RunningStats, x: jnp.ndarray) -> RunningStats:
    """Merges a new batch x (any shape, flattened) into stats using Chan's parallel-variance formula."""
    # 1. Treat x as one flat batch and get its own mean/variance/size.
    x = x.ravel()
    batch_mean, batch_var, batch_count = x.mean(), x.var(), x.size

    # 2. Combine counts and shift the running mean toward whichever side (old stats vs.
    #    new batch) has more samples.
    delta = batch_mean - stats.mean
    total_count = stats.count + batch_count
    new_mean = stats.mean + delta * batch_count / total_count

    # 3. Combine variances: each side's own spread plus a cross term that accounts for
    #    the two sides having had different means before merging.
    combined_m2 = (
        stats.var * stats.count + batch_var * batch_count + delta**2 * stats.count * batch_count / total_count
    )
    return RunningStats(mean=new_mean, var=combined_m2 / total_count, count=total_count)


def normalize_with_stats(stats: RunningStats, x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Standardizes x with the running mean/std."""
    return (x - stats.mean) / jnp.sqrt(stats.var + eps)
