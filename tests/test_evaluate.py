from placax.core import reset  # noqa: F401  must precede jax imports
from placax.types import EnvParams  # noqa: F401
from placax_agents.ops.evaluate import evaluate  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.policy.architectures.cnn import CNNActorCritic  # noqa: F401

import jax.numpy as jnp
from jax import random


def _toy_setup():
    params = EnvParams(grid=8, n_macros=4)
    sizes_array = jnp.array([[2.0, 2.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    padded_pin_idx = jnp.array([[0, 1]])
    padded_pin_offset = jnp.zeros((1, 2, 2))
    valid_mask = jnp.array([[True, True]])
    return params, sizes_array, padded_pin_idx, padded_pin_offset, valid_mask


def test_evaluate_places_every_macro() -> None:
    params, sizes_array, padded_pin_idx, padded_pin_offset, valid_mask = _toy_setup()
    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(init_key, obs0)

    positions, real_hpwl = evaluate(
        variables, policy.apply, params, sizes_array, 1.0,
        padded_pin_idx, padded_pin_offset, valid_mask,
    )
    assert not (positions == -1).any()
    assert jnp.isfinite(real_hpwl)


def test_evaluate_extra_illegal_fn_restricts_actions() -> None:
    # n_macros=1 sidesteps occupancy from an earlier macro conflicting
    # with the forced cell on a later step.
    params = EnvParams(grid=8, n_macros=1)
    sizes_array = jnp.array([[1.0, 1.0]])
    padded_pin_idx = jnp.zeros((1, 1), dtype=jnp.int32)  # one padding-only net, no real pins
    padded_pin_offset = jnp.zeros((1, 1, 2))
    valid_mask = jnp.zeros((1, 1), dtype=bool)
    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(init_key, obs0)

    def only_one_legal_cell(obs):
        illegal = jnp.ones((params.grid, params.grid), dtype=bool)
        return illegal.at[1, 1].set(False)

    positions, _real_hpwl = evaluate(
        variables, policy.apply, params, sizes_array, 1.0,
        padded_pin_idx, padded_pin_offset, valid_mask, extra_illegal_fn=only_one_legal_cell,
    )
    assert positions[0].tolist() == [1, 1]


def test_evaluate_is_deterministic() -> None:
    # Greedy (argmax), no sampling - same policy, same result every time.
    params, sizes_array, padded_pin_idx, padded_pin_offset, valid_mask = _toy_setup()
    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(params), params, sizes_array)
    variables = policy.init(init_key, obs0)

    positions1, hpwl1 = evaluate(
        variables, policy.apply, params, sizes_array, 1.0,
        padded_pin_idx, padded_pin_offset, valid_mask,
    )
    positions2, hpwl2 = evaluate(
        variables, policy.apply, params, sizes_array, 1.0,
        padded_pin_idx, padded_pin_offset, valid_mask,
    )
    assert (positions1 == positions2).all()
    assert hpwl1 == hpwl2
