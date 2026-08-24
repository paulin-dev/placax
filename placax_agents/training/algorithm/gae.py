"""Generalized Advantage Estimation - turns a rollout's rewards and
values into advantages and returns, computed backwards through the
trajectory via lax.scan (a genuine sequential recurrence, not a
candidate for vmap the way wiremask's candidates were)."""
import jax
import jax.numpy as jnp


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    next_value: jax.Array,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[jax.Array, jax.Array]:
    """Returns (advantages, returns). next_value is the bootstrap value
    after the last recorded step - irrelevant here in practice, since
    done=True on that step already zeroes its contribution, but kept for
    correctness if a future non-terminal rollout ever needs it."""

    def scan_fn(gae: jax.Array, transition: tuple) -> tuple[jax.Array, jax.Array]:
        reward, value, next_val, done = transition
        delta = reward + gamma * next_val * (1 - done) - value
        gae = delta + gamma * lam * (1 - done) * gae
        return gae, gae

    next_values = jnp.concatenate([values[1:], next_value[None]])
    _, advantages = jax.lax.scan(
        scan_fn,
        jnp.array(0.0),
        (rewards, values, next_values, dones.astype(jnp.float32)),
        reverse=True,
    )
    returns = advantages + values
    return advantages, returns
