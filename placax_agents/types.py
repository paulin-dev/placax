"""Agent-side type contracts: AlgorithmFn (a policy's apply) and StateFn (an observation builder)."""
from typing import Callable

import jax

AlgorithmFn = Callable[..., tuple[jax.Array, jax.Array]]
"""(variables, obs) -> (action_logits (grid_x, grid_y), value ()); obs must carry "canvas" and "current_macro_size"."""

StateFn = Callable[..., dict]
"""(state, params, sizes_array) -> observation dict; must carry "canvas" and "current_macro_size" (REAL-unit size)."""

ExtraIllegalFn = Callable[[dict], jax.Array]
"""obs -> (grid_x, grid_y) bool extra illegal-cell mask, OR'd into legality in legal_action_logits."""
