from placax_agents.training.algorithm.normalize import normalize_advantages  # noqa: F401  must precede jax imports

import jax.numpy as jnp


def test_normalize_advantages_hand_example() -> None:
    adv = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = normalize_advantages(adv)
    assert abs(float(result.mean())) < 1e-5
    assert abs(float(result.std()) - 1.0) < 1e-4


def test_normalize_advantages_preserves_order() -> None:
    adv = jnp.array([5.0, 1.0, 3.0])
    result = normalize_advantages(adv)
    assert result[1] < result[2] < result[0]


def test_normalize_advantages_constant_input_no_nan() -> None:
    # std=0 for a constant array - the eps term must prevent a NaN from
    # dividing by zero.
    adv = jnp.array([7.0, 7.0, 7.0])
    result = normalize_advantages(adv)
    assert jnp.isfinite(result).all()
