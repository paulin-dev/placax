"""One network, applied once per macro. The same weights for every macro in the design.

**Why sharing is the design and not a shortcut.** A separate network per macro would mean 128
policies on adaptec1's budgeted run, 543 on the full design, none of them transferable to
bigblue1, and each trained on the experience of exactly one block. One shared policy has a single
set of weights, works for any number of macros, gets every macro's experience as training data,
and can be evaluated on a design it never saw - which is the only claim in this line of work that
direct optimization cannot match.

**How "per macro" is implemented: by doing nothing.** `nn.Dense` maps `(..., in) -> (..., out)`, so
handing it a `(n_macros, n_features)` observation already applies the same weights to every row
independently. There is no `vmap` and no batching machinery, and more importantly no way for one
macro's features to leak into another's action - the only channel between macros is the shared
global summary in the `m2` view, and the placement itself.

**tanh, not relu.** The gradient of the placement objective flows back through these activations
over a whole window of steps; relu's dead half would silently zero a macro's learning signal for
as long as its pre-activation stayed negative.

**The action is bounded before it is applied.** `max_step * tanh(raw)` is a displacement of at most
`max_step` grid cells in each axis, so exploration noise cannot fling a macro across the canvas and
`log_std` can be learned rather than scheduled. The noise is added BEFORE the tanh, which keeps the
sample a differentiable (reparameterized) function of the parameters - the property SHAC needs and
a categorical action cannot provide.
"""
from placax import _device  # noqa: F401  must precede jax imports

import jax
import jax.numpy as jnp
from flax import linen as nn


class SharedMacroPolicy(nn.Module):
    """`(n_macros, n_features) -> (n_macros, 2)` mean displacement, plus one learned noise width."""

    features: tuple[int, ...] = (64, 64)
    init_log_std: float = -1.5
    """Starting exploration width, as a fraction of `max_step`: exp(-1.5) is about 0.22 of a step.

    Narrow on purpose. The episode starts from the best placement this project has, so the useful
    first moves are small corrections; a wide start would spend the early iterations learning to
    undo its own noise."""

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        hidden = obs
        for width in self.features:
            hidden = nn.tanh(nn.Dense(width)(hidden))
        # Zero-initialized output layer: the policy starts by asking every macro to stay where it
        # is, so iteration 0 scores exactly the warm start plus noise rather than a random shove.
        mean_raw = nn.Dense(2, kernel_init=nn.initializers.zeros)(hidden)
        # float32 explicitly. `jnp.full((2,), -1.5)` under this project's x64 (placax/_device.py)
        # is a WEAKLY typed float64: it behaves as float32 on the first iteration and as real
        # float64 after the optimizer touches it, which turned the placement into float64 at
        # iteration 2 and broke the rollout's scan carry. Every other parameter here is float32
        # because Dense's param_dtype says so; this one has to say so itself.
        log_std = self.param(
            "log_std", lambda _key: jnp.full((2,), self.init_log_std, dtype=jnp.float32)
        )
        return mean_raw, log_std


def displacements(
    mean_raw: jax.Array,
    log_std: jax.Array,
    key: jax.Array,
    max_step: float,
    stochastic: bool = True,
) -> jax.Array:
    """Turns the policy's output into a bounded displacement in grid cells.

    Stochastic during training (reparameterized, so the gradient reaches `log_std` too) and the
    mean during evaluation - an evaluated placement should be the policy's actual recommendation,
    not one sample of it.
    """
    raw = mean_raw
    if stochastic:
        # `dtype` explicitly: this project enables JAX's x64 (placax/_device.py, for MaskPlace's
        # own float64 logit cast), so an unqualified `normal` returns float64 and would promote
        # the whole placement - which `lax.scan` rejects as a carry whose type changed.
        raw = raw + jnp.exp(log_std) * jax.random.normal(key, mean_raw.shape, dtype=mean_raw.dtype)
    return max_step * jnp.tanh(raw)
