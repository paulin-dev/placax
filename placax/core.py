"""reset()/step() - the shared environment kernel every agent drives."""
from placax import _device  # noqa: F401  must precede jax imports
from placax.action_space import DISCRETE_GRID, ActionSpace
from placax.types import EnvParams, EnvState, RewardFn

import jax
import jax.numpy as jnp


def reset(
    params: EnvParams,
    initial_positions: jax.Array | None = None,
    action_space: ActionSpace = DISCRETE_GRID,
) -> EnvState:
    """Starts an episode, optionally warm-started with a prefix of macros already placed."""
    return action_space.reset(params, initial_positions)


def step(
    state: EnvState,
    action: jax.Array,
    reward_fn: RewardFn,
    params: EnvParams,
    action_space: ActionSpace = DISCRETE_GRID,
) -> tuple[EnvState, jax.Array, jax.Array]:
    """Applies one action and asks reward_fn for the reward, returning (new_state, reward, done).

    What an action IS, what it changes, and when the episode ends all belong to `action_space` -
    see placax/action_space.py. They used to be one line here, which made the sequential-
    constructive paradigm look like the only paradigm in a project built to compare several.
    The default is that same behaviour, so nothing that doesn't ask for another space can tell.
    """
    # 1. Apply the action, however this space defines that.
    new_state = action_space.apply(state, action, params)
    # 2. Episode termination is the space's call too: a constructive space ends when every macro
    #    is placed, a perturbation space when its move budget runs out.
    done = action_space.done(new_state, params)
    # 3. Work out which macros were/are placed (x >= 0), needed by reward_fn but not
    #    re-derivable after a real-unit conversion, so we pass it explicitly.
    old_placed = state.positions[:, 0] >= 0
    new_placed = new_state.positions[:, 0] >= 0
    # 4. Delegate the actual reward shaping (sparse vs. dense) to the caller's reward_fn.
    reward = reward_fn(state.positions, new_state.positions, old_placed, new_placed)
    return new_state, reward, done


def replay(
    positions: jax.Array,
    reward_fn: RewardFn,
    params: EnvParams,
    n_placed: int = 0,
    action_space: ActionSpace = DISCRETE_GRID,
) -> jax.Array:
    """Drives the kernel through an already-decided placement, returning the episode's total reward.

    Every agent hands the runner a PLACEMENT, never a score - so scoring one against the run's
    configured reward means replaying it through the same `step()` a policy would have driven,
    rather than reimplementing the reward's own accumulation somewhere else. Sparse and dense
    reward functions both come out right, since the sum over the episode is what they agree on.

    This is also the population-method entry point the design has always claimed and never had:
    `jax.vmap(replay, in_axes=(0, None, None, None))` scores a whole GA population of
    pre-committed action sequences against the identical kernel, with no new machinery. `step()`
    cannot tell the difference between an action sampled from a policy and one read out of a
    genome, which was the point.

    `n_placed` is the environment's warm-start prefix: those rows are kept as given and replay
    starts after them. Constructive spaces only - a placement is a sequence of appends there, and
    is not a sequence of anything under a space whose actions move committed macros.
    """
    # Rebuild the starting state: the warm-start prefix as it was, everything after it unplaced.
    initial_positions = positions.at[n_placed:].set(-1)

    def scan_step(carry, action):
        state, total = carry
        state, reward, _done = step(state, action, reward_fn, params, action_space)
        return (state, total + reward), None

    start = (reset(params, initial_positions, action_space), jnp.array(0.0))
    (_final_state, total), _ = jax.lax.scan(scan_step, start, positions[n_placed:])
    return total


def random_action(
    key: jax.Array, params: EnvParams, action_space: ActionSpace = DISCRETE_GRID
) -> jax.Array:
    """Trivial action sampler for smoke tests. What an action LOOKS like belongs to the space."""
    return action_space.random_action(key, params)
