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
    # 4. Delegate the actual reward shaping (sparse vs. dense) to the caller's reward_fn - with
    #    the placement's ORIENTATIONS when the space produces any. A turned macro sits at a
    #    different center and its pins are somewhere else, so a reward computed without them
    #    scores a placement that was never made: measured at 21.91 vs 23.91 real HPWL for one
    #    placement at two orientation assignments, while the reward returned the same number for
    #    both. Passed only when the state carries them, so every un-oriented run calls the same
    #    four-argument reward it always did, bit for bit - see placax/types.py's RewardFn.
    if new_state.orientations is None:
        reward = reward_fn(state.positions, new_state.positions, old_placed, new_placed)
    else:
        reward = reward_fn(state.positions, new_state.positions, old_placed, new_placed,
                           new_state.orientations)
    return new_state, reward, done


def replay(
    positions: jax.Array,
    reward_fn: RewardFn,
    params: EnvParams,
    n_placed: int = 0,
    action_space: ActionSpace = DISCRETE_GRID,
    orientations: jax.Array | None = None,
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
    is not a sequence of anything under a space whose actions move committed macros, which
    `action_space.encode` refuses rather than silently mis-replays.

    `orientations` are the turns that came with the placement, for a space that has them. The
    space turns the pair back into actions: feeding the bare position rows in worked only for a
    space whose action IS a grid cell, and under `oriented_grid` the y coordinate was read as the
    orientation.
    """
    if not action_space.constructive:
        raise ValueError(
            f"replay() re-drives a placement through step() one action at a time, which the "
            f"{action_space.name!r} action space cannot express - its actions move macros that "
            f"are already placed, so a finished placement carries no episode to replay. Score it "
            f"against the placement its episode started from instead; "
            f"placax_agents.experiment.run.episode_return does that for whichever space a run "
            f"configured."
        )
    # Rebuild the starting state: the warm-start prefix as it was, everything after it unplaced.
    initial_positions = positions.at[n_placed:].set(-1)
    actions = action_space.encode(positions, orientations)

    def scan_step(carry, action):
        state, total = carry
        state, reward, _done = step(state, action, reward_fn, params, action_space)
        return (state, total + reward), None

    start = (reset(params, initial_positions, action_space), jnp.array(0.0))
    (_final_state, total), _ = jax.lax.scan(scan_step, start, actions[n_placed:])
    return total


def random_action(
    key: jax.Array, params: EnvParams, action_space: ActionSpace = DISCRETE_GRID
) -> jax.Array:
    """Trivial action sampler for smoke tests. What an action LOOKS like belongs to the space."""
    return action_space.random_action(key, params)
