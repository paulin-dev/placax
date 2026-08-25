"""reset()/step() - the shared environment kernel every agent drives."""
from placax import _device  # noqa: F401  must precede jax imports
from placax.types import EnvParams, EnvState, RewardFn

import jax
import jax.numpy as jnp
from jax import random


def reset(params: EnvParams, initial_positions: jax.Array | None = None) -> EnvState:
    """Starts an episode. initial_positions, if given, warm-starts with
    the first k macros placed (a prefix only) and the rest at (-1, -1)."""
    if initial_positions is None:
        initial_positions = jnp.full((params.n_macros, 2), -1)
    n_placed = (initial_positions[:, 0] >= 0).sum()
    return EnvState(positions=initial_positions, step=n_placed)


def step(
    state: EnvState, action: jax.Array, reward_fn: RewardFn, params: EnvParams
) -> tuple[EnvState, jax.Array, jax.Array]:
    """Places the next macro and calls reward_fn (see placax.types.RewardFn);
    sparse vs. dense reward is entirely reward_fn's choice. Returns
    (new_state, reward, done)."""
    positions = state.positions.at[state.step].set(action)  # write this macro's chosen cell
    new_state = EnvState(positions=positions, step=state.step + 1)
    done = new_state.step == params.n_macros
    old_placed = state.positions[:, 0] >= 0
    new_placed = positions[:, 0] >= 0
    reward = reward_fn(state.positions, positions, old_placed, new_placed)
    return new_state, reward, done


def random_action(key: jax.Array, params: EnvParams) -> jax.Array:
    """Trivial action sampler for smoke tests; x/y sampled independently
    so it's correct on non-square canvases too."""
    x_key, y_key = random.split(key)
    x = random.randint(x_key, (), 0, params.grid_x)
    y = random.randint(y_key, (), 0, params.effective_grid_y)
    return jnp.array([x, y])
