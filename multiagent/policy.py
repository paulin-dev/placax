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


def step_bound(progress: jax.Array, max_step: float, max_step_start: float | None = None):
    """The largest move allowed at `progress` (0 at the first step, ->1 at the last), in cells.

    Constant `max_step` by default. With `max_step_start`, it shrinks geometrically from that to
    `max_step` over the episode - big jumps first, to reach a different arrangement, then small
    corrections. Geometric for the reason the density ramp is: the useful range spans orders of
    magnitude.
    """
    if max_step_start is None or max_step_start == max_step:
        return max_step
    return max_step_start * (max_step / max_step_start) ** progress


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


def alignment_matrix(connection_weights) -> jax.Array:
    """Row-normalized connection weights: row i is macro i's partners, summing to 1 (or 0)."""
    weights = jnp.asarray(connection_weights, dtype=jnp.float32)
    total = weights.sum(axis=1, keepdims=True)
    return jnp.where(total > 0, weights / jnp.maximum(total, 1e-9), 0.0)


def make_act(policy: nn.Module, view_fn, config: dict, connection_weights):
    """The whole decision rule, built from a run's settings - so train, transfer and visualize
    move macros identically. `config` is a run's args (a manifest's `args` dict works as is).

    On top of the network's own displacement, two swarm rules, both off by default:

    * **alignment** (`align` = a in [0, 1]) - Boids' third rule, the one placement never uses. A
      macro's move becomes `(1 - a) * own + a * (weighted mean of its partners' moves)`, a convex
      combination, so it stays inside the step bound. At a near 1 a connected cluster moves as one
      school: its internal wires stay short while the group as a whole travels, which is the kind
      of long move a single macro's gradient never takes. A macro with no partners keeps its own.
    * **asynchronous updates** (`update_prob` = p) - each macro moves on a given step with
      probability p, as in Growing Neural Cellular Automata. Breaks the lockstep symmetry behind
      two partners chasing each other. Also applied in evaluation (with a fixed key there), because
      the rule was trained that way.

    Returns `act(variables, positions, parts, progress, key, stochastic) -> deltas`.
    """
    max_step = float(config["max_step"])
    start = config.get("max_step_start")
    start = None if start in (None, "None") else float(start)
    align = float(config.get("align", 0.0) or 0.0)
    update_prob = float(config.get("update_prob", 1.0) or 1.0)
    partners = alignment_matrix(connection_weights)
    connected = (partners.sum(axis=1, keepdims=True) > 0)

    def act(variables, positions, parts, progress, key, stochastic):
        noise_key, update_key = jax.random.split(key)
        mean_raw, log_std = policy.apply(variables, view_fn(positions, parts, progress))
        bound = step_bound(progress, max_step, start)
        deltas = displacements(mean_raw, log_std, noise_key, bound, stochastic)
        if align > 0.0:
            shared = partners @ deltas
            deltas = jnp.where(connected, (1.0 - align) * deltas + align * shared, deltas)
        if update_prob < 1.0:
            moving = jax.random.bernoulli(update_key, update_prob, (deltas.shape[0], 1))
            deltas = deltas * moving
        return deltas.astype(positions.dtype)

    return act
