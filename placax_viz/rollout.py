"""Step-by-step placement history for a trained policy, for animating a rollout.

Drives the real `step()` rather than writing the transition itself. It used to do the latter -
`positions.at[state.step].set(action)`, copied out of the kernel - which is precisely what
`ops/evaluate.py` warns against ("a second copy of that line here is how the 'one shared kernel'
property quietly stops being true"). It had already drifted in one way that mattered: it ignored
the warm start, so an animation of a warm-started run began from an empty canvas.
"""
from placax.action_space import DISCRETE_GRID, ActionSpace
from placax.core import reset, step
from placax_agents.policy.action import legal_action_logits
from placax_agents.policy.observation import observation
from placax_agents.policy.scale import to_grid_units
from placax_agents.types import AlgorithmFn, ExtraIllegalFn, StateFn

import jax.numpy as jnp
import numpy as np


def collect_placement_history(
    variables,
    policy_apply_fn: AlgorithmFn,
    params,
    sizes_array,
    cell_size: float,
    state_fn: StateFn = observation,
    extra_illegal_fn: ExtraIllegalFn | None = None,
    initial_positions=None,
    n_placed: int = 0,
    action_space: ActionSpace = DISCRETE_GRID,
) -> list:
    """Greedily places every remaining macro, one positions array per step from start to final layout."""
    state = reset(params, initial_positions, action_space)
    history = [np.asarray(state.positions)]
    for _ in range(action_space.episode_length(params, n_placed)):
        obs = state_fn(state, params, sizes_array)
        logits, _value = policy_apply_fn(variables, obs)

        macro_size = to_grid_units(obs["current_macro_size"], cell_size)
        extra_illegal = extra_illegal_fn(obs) if extra_illegal_fn is not None else None
        masked_logits = legal_action_logits(logits, obs["canvas"], params, macro_size, extra_illegal)

        flat_idx = jnp.argmax(masked_logits.ravel())
        grid_y = masked_logits.shape[1]
        action = jnp.array([flat_idx // grid_y, flat_idx % grid_y])

        # The kernel applies it, so an animation shows exactly what a rollout would have done.
        state, _reward, _done = step(state, action, _no_reward, params, action_space)
        history.append(np.asarray(state.positions))
    return history


def _no_reward(_old_positions, _new_positions, _old_placed, _new_placed):
    """A history is about positions; the per-step reward is unused and never computed."""
    return jnp.array(0.0)
