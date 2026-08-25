"""reset()/step() - the shared environment kernel every agent drives."""
from placax import _device  # noqa: F401  must precede jax imports
from placax.types import EnvParams, EnvState, RewardFn

import jax
import jax.numpy as jnp
from jax import random


def reset(params: EnvParams, initial_positions: jax.Array | None = None) -> EnvState:
    """Starts an episode, optionally warm-started with a prefix of macros already placed."""
    # Default: nothing placed yet, every macro sits at the (-1, -1) sentinel.
    if initial_positions is None:
        initial_positions = jnp.full((params.n_macros, 2), -1)
    # Count how many macros the warm-start already placed, so step() resumes from there.
    n_placed = (initial_positions[:, 0] >= 0).sum()
    return EnvState(positions=initial_positions, step=n_placed)


def step(
    state: EnvState, action: jax.Array, reward_fn: RewardFn, params: EnvParams
) -> tuple[EnvState, jax.Array, jax.Array]:
    """Places the next macro and asks reward_fn for the reward, returning (new_state, reward, done)."""
    # 1. Write the chosen cell into this step's macro row, advancing the step counter.
    positions = state.positions.at[state.step].set(action)
    new_state = EnvState(positions=positions, step=state.step + 1)
    # 2. Episode ends once every macro has been placed.
    done = new_state.step == params.n_macros
    # 3. Work out which macros were/are placed (x >= 0), needed by reward_fn but not
    #    re-derivable after a real-unit conversion, so we pass it explicitly.
    old_placed = state.positions[:, 0] >= 0
    new_placed = positions[:, 0] >= 0
    # 4. Delegate the actual reward shaping (sparse vs. dense) to the caller's reward_fn.
    reward = reward_fn(state.positions, positions, old_placed, new_placed)
    return new_state, reward, done


def random_action(key: jax.Array, params: EnvParams) -> jax.Array:
    """Trivial action sampler for smoke tests, sampling x/y independently for non-square canvases."""
    x_key, y_key = random.split(key)
    x = random.randint(x_key, (), 0, params.grid_x)
    y = random.randint(y_key, (), 0, params.effective_grid_y)
    return jnp.array([x, y])
