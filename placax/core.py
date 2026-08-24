"""reset() / step() - the one shared kernel every agent drives."""
from placax import _device  # noqa: F401

import jax
import jax.numpy as jnp
from jax import random

from placax.types import EnvParams, EnvState, RewardFn


def reset(params: EnvParams, initial_positions: jax.Array | None = None) -> EnvState:
    """Start an episode. Defaults to nothing placed. For a warm start,
    pass initial_positions with the first k macros already placed, in
    index order, and the rest at the (-1, -1) sentinel - step() always
    places macros in index order, so only a *prefix* of pre-placed
    macros is supported here, not arbitrary gaps."""
    if initial_positions is None:
        initial_positions = jnp.full((params.n_macros, 2), -1)
    n_placed = (initial_positions[:, 0] >= 0).sum()
    return EnvState(positions=initial_positions, step=n_placed)


def step(
    state: EnvState, action: jax.Array, reward_fn: RewardFn, params: EnvParams
) -> tuple[EnvState, jax.Array, jax.Array]:
    """Place the next macro; reward only fires once every macro is placed."""
    idx = state.step
    positions = state.positions.at[idx].set(action)
    new_state = EnvState(positions=positions, step=idx + 1)
    done = new_state.step == params.n_macros
    reward = jax.lax.cond(done, lambda: reward_fn(positions), lambda: 0.0)
    return new_state, reward, done


def random_action(key: jax.Array, params: EnvParams) -> jax.Array:
    """A trivial action sampler, for smoke-testing the kernel alone.
    Samples x and y from their own ranges independently - correct even
    for a non-square canvas (params.grid_x != params.effective_grid_y)."""
    x_key, y_key = random.split(key)
    x = random.randint(x_key, (), 0, params.grid_x)
    y = random.randint(y_key, (), 0, params.effective_grid_y)
    return jnp.array([x, y])
