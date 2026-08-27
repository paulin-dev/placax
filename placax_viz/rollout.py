"""Step-by-step placement history for a trained policy, for animating a rollout
(placax_viz.animation). Mirrors placax_agents.ops.evaluate.evaluate's greedy rollout, but as a
plain Python loop (not jax.lax.scan) so every intermediate EnvState.positions is kept, not just
the final one - too slow to train with, fine for building an animation of one trained policy."""
from placax.core import reset
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
) -> list:
    """Greedily places every macro (argmax over legal cells), returning one (n_macros, 2) grid-unit
    positions array per step - history[0] is all-unplaced, history[-1] is the final layout."""
    state = reset(params)
    history = [np.asarray(state.positions)]
    for _ in range(params.n_macros):
        obs = state_fn(state, params, sizes_array)
        logits, _value = policy_apply_fn(variables, obs)

        macro_size = to_grid_units(obs["current_macro_size"], cell_size)
        extra_illegal = extra_illegal_fn(obs) if extra_illegal_fn is not None else None
        masked_logits = legal_action_logits(logits, obs["canvas"], params, macro_size, extra_illegal)

        flat_idx = jnp.argmax(masked_logits.ravel())
        grid_y = masked_logits.shape[1]
        action = jnp.array([flat_idx // grid_y, flat_idx % grid_y])

        positions = state.positions.at[state.step].set(action)
        state = state.replace(positions=positions, step=state.step + 1)
        history.append(np.asarray(state.positions))
    return history
