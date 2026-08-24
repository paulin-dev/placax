"""Shared type contracts for placax_agents: algorithm_fn and state_fn,
the two swappable axes placax's own core kernel has no formal
dependency on (reward_fn lives in placax.types instead, since step()
itself consumes it directly - these two are entirely agent-side)."""
from typing import Callable

import jax

AlgorithmFn = Callable[..., tuple[jax.Array, jax.Array]]
"""(variables, canvas) -> (action_logits, value) - a policy's real apply()
contract, e.g. CNNActorCritic.apply. Loose on input args (Flax module
apply() signatures aren't uniformly Callable-typeable), strict on the
output shape every caller actually depends on."""

StateFn = Callable[..., dict]
"""(state, params, sizes_array) -> observation dict, e.g. observation()
itself. Returns a dict, not a fixed type, since different state_fn
implementations can expose different keys (canvas for image-based
policies, raw positions/sizes for anything else - see observation.py)."""
