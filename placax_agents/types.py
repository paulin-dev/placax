"""Agent-side type contracts: AlgorithmFn (a policy's apply) and StateFn
(an observation builder) - the two swappable axes the core kernel takes
no formal dependency on."""
from typing import Callable

import jax

AlgorithmFn = Callable[..., tuple[jax.Array, jax.Array]]
"""(variables, obs) -> (action_logits, value), e.g. CNNActorCritic.apply."""

StateFn = Callable[..., dict]
"""(state, params, sizes_array) -> observation dict, e.g. observation()."""
