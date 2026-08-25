"""Agent-side type contracts: AlgorithmFn (a policy's apply) and StateFn
(an observation builder)."""
from typing import Callable

import jax

AlgorithmFn = Callable[..., tuple[jax.Array, jax.Array]]
"""(variables, obs) -> (action_logits (grid_x, grid_y) float, value ()
float), e.g. CNNActorCritic.apply. obs["canvas"] and
obs["current_macro_size"] must be present regardless of policy, since
legal_action_logits reads them."""

StateFn = Callable[..., dict]
"""(state, params, sizes_array) -> observation dict, e.g. observation().
Must include "canvas" ((grid_x, grid_y) bool, placed footprints) and
"current_macro_size" ((2,) float, REAL-unit size of the macro being
placed - convert with policy.scale.to_grid_units before
legal_action_logits). Other keys are policy-specific conventions, e.g.
"wiremask" from policy.observation.make_wiremask_observation."""

ExtraIllegalFn = Callable[[dict], jax.Array]
"""obs -> (grid_x, grid_y) bool, an extra illegal-cell cutoff OR'd into
legality in legal_action_logits (see placax.extras.masks.quality_mask).
Optional on collect_rollout/ppo_loss/evaluate; None means legality-only
masking."""
