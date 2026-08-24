from placax.core import reset  # noqa: F401  must precede jax imports
from placax.types import EnvParams  # noqa: F401
from placax_agents.policy import CNNPolicy, legal_action_logits, sample_action  # noqa: F401

import jax
import jax.numpy as jnp
from jax import random


def test_cnn_policy_output_shape() -> None:
    canvas = jnp.zeros((8, 8), dtype=bool)
    policy = CNNPolicy()
    variables = policy.init(random.PRNGKey(0), canvas)
    logits = policy.apply(variables, canvas)
    assert logits.shape == (8, 8)


def test_legal_action_logits_masks_occupied_and_out_of_bounds() -> None:
    params = EnvParams(grid=4, n_macros=2)
    state = reset(params)
    state = state.replace(positions=state.positions.at[0].set(jnp.array([0, 0])))
    logits = jnp.zeros((4, 4))

    masked = legal_action_logits(logits, state, params, macro_size=(1, 1))
    assert masked[0, 0] == -jnp.inf  # occupied
    assert masked[1, 1] == 0.0  # legal, unmasked


def test_legal_action_logits_masks_footprint_that_would_overflow() -> None:
    params = EnvParams(grid=4, n_macros=1)
    state = reset(params)
    logits = jnp.zeros((4, 4))

    masked = legal_action_logits(logits, state, params, macro_size=(2, 2))
    assert masked[3, 3] == -jnp.inf  # a 2x2 macro can't start at the last row/col
    assert masked[0, 0] == 0.0  # fits fine


def test_sample_action_only_picks_legal_cells() -> None:
    logits = jnp.full((4, 4), -jnp.inf)
    logits = logits.at[2, 3].set(0.0)  # exactly one legal cell

    for seed in range(10):
        action = sample_action(random.PRNGKey(seed), logits)
        assert action.tolist() == [2, 3]
