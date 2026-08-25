"""Agent-side type contracts: AlgorithmFn (a policy's apply) and StateFn
(an observation builder)."""
from typing import Callable

import jax

AlgorithmFn = Callable[..., tuple[jax.Array, jax.Array]]
"""(variables, obs) -> (action_logits, value), e.g. CNNActorCritic.apply.
obs is whatever your StateFn returns (see below) - a policy only needs to
read the keys it actually uses, but obs["canvas"] and
obs["current_macro_size"] must still be present, since
placax_agents.policy.action.legal_action_logits reads them regardless of
policy. action_logits: (grid_x, grid_y) float. value: () float scalar."""

StateFn = Callable[..., dict]
"""(state, params, sizes_array) -> observation dict, e.g. observation().
Two keys are required by the training/eval loops themselves, not just the
default policy - omitting them breaks legal_action_logits(), regardless
of what policy you pair this with:
  "canvas": (grid_x, grid_y) bool, already-placed footprints.
  "current_macro_size": (2,) float, REAL-unit (width, height) of the
    macro being placed - convert with policy.scale.to_grid_units(...,
    cell_size) before passing to legal_action_logits, as rollout/evaluate do.
Any other keys are yours to add for a custom policy (observation()'s
positions/sizes_array/placed_mask/step/lookahead_sizes are conventions,
not requirements - as is "wiremask", added by
policy.observation.make_wiremask_observation)."""
