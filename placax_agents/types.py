"""Agent-side type contracts: AlgorithmFn (a policy's apply) and StateFn (an observation builder)."""
from typing import Callable

import jax

AlgorithmFn = Callable[..., tuple[jax.Array, jax.Array]]
"""(variables, obs) -> (action_logits (grid_x, grid_y), value ()); obs must carry "canvas" and "current_macro_size"."""

StateFn = Callable[..., dict]
"""(state, params, sizes_array) -> observation dict; must carry "canvas" and "current_macro_size" (REAL-unit size)."""

ExtraIllegalFn = Callable[[dict], jax.Array]
"""obs -> (grid_x, grid_y) bool extra illegal-cell mask, OR'd into legality in legal_action_logits."""

InitFn = Callable[[jax.Array], "jax.Array | None"]
"""key -> (n_macros, 2) int grid positions with unplaced macros at the -1 sentinel, or None for
an empty canvas. Resolved ONCE per run (not per episode) so the number of macros the agent still
has to place is a static shape, and so "the initial placement" is a property of the environment
rather than something that varies inside it."""
